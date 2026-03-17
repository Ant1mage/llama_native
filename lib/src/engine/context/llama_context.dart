import 'dart:async';
import 'dart:ffi';
import 'dart:isolate';
import 'package:ffi/ffi.dart';

import 'package:llama_native/src/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/engine/backend/llama_backend.dart';
import 'package:llama_native/src/engine/backend/llama_backend_config.dart';
import 'package:llama_native/src/engine/model/llama_model.dart';
import 'package:llama_native/src/engine/context/inference_config.dart';
import 'package:llama_native/src/engine/context/token_generation.dart';
import 'package:llama_native/src/engine/sampling/sampling_config.dart';
import 'package:llama_native/src/engine/cache/kv_cache_manager.dart';
import 'package:llama_native/src/log/logger.dart';
import 'package:llama_native/src/utils/disposable.dart';
import 'package:llama_native/src/engine/exceptions/llama_exceptions.dart';
import 'package:llama_native/src/engine/grammar/grammar.dart';

/// 推理执行引擎
class LlamaContext with Disposable {
  final LlamaModel _model;
  final InferenceConfig _config;
  final Logger _logger;

  Pointer<bindings.llama_context>? _ctxPtr;
  Pointer<bindings.llama_sampler>? _samplerChain;
  KVCacheManager? _kvCache;
  bool _disposed = false;
  int _nPast = 0;

  /// 私有构造函数
  LlamaContext._(this._model, this._config) : _logger = Logger('LlamaContext');

  factory LlamaContext.create(LlamaModel model, InferenceConfig config) {
    final context = LlamaContext._(model, config);
    context._initialize();
    return context;
  }

  void _initialize() {
    if (_ctxPtr != null) {
      throw StateError('Context already initialized');
    }

    _logger.info(
      'Creating context: n_ctx=${_config.nCtx}, n_batch=${_config.nBatch}, n_gpu_layers=${_config.nGpuLayers}',
    );

    final backend = LlamaBackend.createWithConfig(LlamaBackendConfig.forGpuLayers(_config.nGpuLayers));

    final ctxParams = backend.getContextParams(
      nCtx: _config.nCtx,
      nBatch: _config.nBatch,
      nUBatch: _config.nUBatch,
      nThreads: _config.nThreads,
    );

    final ptr = bindings.llama_init_from_model(_model.handle, ctxParams);
    if (ptr == nullptr) {
      throw LlamaContextInitException('llama_init_from_model returned null');
    }
    _ctxPtr = ptr;

    _samplerChain = _buildSamplerChain(_config.sampling);

    _kvCache = KVCacheManager(nCtx: _config.nCtx, ctx: _ctxPtr);

    _logger.info('Context initialized');
  }

  /// 使用 llama.cpp 原生 API 构建 sampler chain
  Pointer<bindings.llama_sampler> _buildSamplerChain(SamplingConfig sampling) {
    final chainParams = bindings.llama_sampler_chain_default_params();
    final chain = bindings.llama_sampler_chain_init(chainParams);

    if (chain == nullptr) {
      throw LlamaContextInitException('Failed to init sampler chain');
    }

    // 按照 llama.cpp 推荐顺序添加 sampler
    // 1. penalties（重复惩罚）
    if (sampling.penaltyRepeat != 1.0 || sampling.frequencyPenalty != 0.0 || sampling.presencePenalty != 0.0) {
      bindings.llama_sampler_chain_add(
        chain,
        bindings.llama_sampler_init_penalties(
          sampling.penaltyLastN,
          sampling.penaltyRepeat,
          sampling.frequencyPenalty,
          sampling.presencePenalty,
        ),
      );
    }

    if (sampling.temperature <= 0.0) {
      // Greedy sampling
      bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_greedy());
    } else {
      // top-k -> min-p -> top-p -> temperature -> distribution
      if (sampling.topK > 0) {
        bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_top_k(sampling.topK));
      }
      if (sampling.minP > 0.0) {
        bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_min_p(sampling.minP, 1));
      }
      if (sampling.topP < 1.0) {
        bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_top_p(sampling.topP, 1));
      }
      bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_temp(sampling.temperature));
      bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_dist(0xFFFFFFFF)); // LLAMA_DEFAULT_SEED
    }

    return chain;
  }

  /// 获取上下文指针
  Pointer<bindings.llama_context> get handle {
    if (_ctxPtr == null || _disposed) {
      throw StateError('Context is disposed');
    }
    return _ctxPtr!;
  }

  /// 获取 KV Cache 管理器
  KVCacheManager get kvCache {
    if (_kvCache == null) {
      throw StateError('KVCache not initialized');
    }
    return _kvCache!;
  }

  /// 当前已处理的 token 数
  int get nPast => _nPast;

  /// 是否已处置
  bool get isDisposed => _disposed;

  /// 流式生成（在独立 Isolate 中运行，避免阻塞主线程/GPU watchdog）
  Stream<TokenGeneration> generateStream(List<int> inputTokens, {int maxTokens = 2048, Grammar? grammar}) async* {
    if (_disposed) throw StateError('Context is disposed');
    if (inputTokens.isEmpty) return;

    final maxPossibleTokens = _config.nCtx - _nPast - inputTokens.length;
    if (maxPossibleTokens <= 0) {
      _logger.error('Context full: nCtx=${_config.nCtx}, nPast=$_nPast, inputTokens=${inputTokens.length}');
      throw LlamaContextInitException('Context is full, cannot generate more tokens. Call reset() to clear context.');
    }
    final actualMaxTokens = maxTokens.clamp(1, maxPossibleTokens);

    if (actualMaxTokens < maxTokens) {
      _logger.warning('maxTokens reduced from $maxTokens to $actualMaxTokens due to context limit');
    }

    final receivePort = ReceivePort();
    Isolate? isolate;

    try {
      final args = _IsolateArgs(
        sendPort: receivePort.sendPort,
        ctxAddress: _ctxPtr!.address,
        samplerAddress: _samplerChain!.address,
        vocabAddress: _model.vocab.address,
        inputTokens: inputTokens,
        maxTokens: actualMaxTokens,
        grammarAddress: grammar?.sampler.address ?? 0,
        nUBatch: _config.nUBatch,
        currentNPast: _nPast,
      );

      isolate = await Isolate.spawn(_inferenceIsolate, args);

      await for (final msg in receivePort) {
        if (msg is _TokenMsg) {
          final tokenText = _model.detokenize([msg.token]);
          _logger.debug('Token: ${msg.token}, Text: "$tokenText", isEnd: ${msg.isEnd}');
          final result = TokenGeneration(token: msg.token, text: tokenText, isEnd: msg.isEnd);
          _nPast = msg.nPast;
          _kvCache?.addProcessed(msg.isEnd ? 0 : 1);
          yield result;
          if (msg.isEnd) break;
        } else if (msg is _ErrorMsg) {
          _logger.error('Inference error: ${msg.message}');
          throw LlamaContextInitException(msg.message);
        } else if (msg == null) {
          break;
        }
      }
    } catch (e) {
      _logger.error('Error in generateStream: $e');
      rethrow;
    } finally {
      receivePort.close();
      if (isolate != null) {
        try {
          isolate.kill(priority: Isolate.immediate);
        } catch (e) {
          _logger.warning('Error killing isolate: $e');
        }
      }
    }
  }

  /// 在 Isolate 内运行的推理入口（static，不捕获 this）
  static void _inferenceIsolate(_IsolateArgs args) {
    final sendPort = args.sendPort;

    try {
      final ctx = Pointer<bindings.llama_context>.fromAddress(args.ctxAddress);
      final sampler = Pointer<bindings.llama_sampler>.fromAddress(args.samplerAddress);
      final vocab = Pointer<bindings.llama_vocab>.fromAddress(args.vocabAddress);
      final grammarSampler = args.grammarAddress != 0
          ? Pointer<bindings.llama_sampler>.fromAddress(args.grammarAddress)
          : nullptr;

      int nPast = args.currentNPast;

      final tokens = args.inputTokens;
      final prefillChunkSize = args.nUBatch;
      var offset = 0;
      while (offset < tokens.length) {
        final end = (offset + prefillChunkSize).clamp(0, tokens.length);
        final chunk = tokens.sublist(offset, end);
        _decodeChunk(ctx, chunk, nPast, sendPort);
        nPast += chunk.length;
        offset = end;
      }

      for (var i = 0; i < args.maxTokens; i++) {
        try {
          int sampledToken;
          if (grammarSampler != nullptr) {
            sampledToken = bindings.llama_sampler_sample(grammarSampler, ctx, -1);
            bindings.llama_sampler_accept(grammarSampler, sampledToken);
          } else {
            sampledToken = bindings.llama_sampler_sample(sampler, ctx, -1);
            bindings.llama_sampler_accept(sampler, sampledToken);
          }

          final isEnd = bindings.llama_token_is_eog(vocab, sampledToken);
          nPast += 1;

          sendPort.send(_TokenMsg(token: sampledToken, isEnd: isEnd, nPast: nPast));

          if (isEnd) break;

          _decodeChunk(ctx, [sampledToken], nPast - 1, sendPort);
        } catch (e) {
          sendPort.send(_ErrorMsg('Error during token generation: ${e.toString()}'));
          return;
        }
      }

      sendPort.send(null);
    } catch (e) {
      sendPort.send(_ErrorMsg('Inference isolate error: ${e.toString()}'));
    }
  }

  /// 静态方法：解码一批 token（在 Isolate 内调用）
  static void _decodeChunk(Pointer<bindings.llama_context> ctx, List<int> tokens, int startPos, SendPort sendPort) {
    if (tokens.isEmpty) return;

    final nTokens = tokens.length;
    final batch = bindings.llama_batch_init(nTokens, 0, 1);

    try {
      for (var i = 0; i < nTokens; i++) {
        batch.token.elementAt(i).value = tokens[i];
        batch.pos.elementAt(i).value = startPos + i;
        batch.n_seq_id.elementAt(i).value = 1;
        batch.seq_id.elementAt(i).value.elementAt(0).value = 0;
        batch.logits.elementAt(i).value = 0;
      }
      batch.logits.elementAt(nTokens - 1).value = 1;
      batch.n_tokens = nTokens;

      final ret = bindings.llama_decode(ctx, batch);
      if (ret < 0) {
        throw LlamaContextInitException('llama_decode failed: $ret');
      } else if (ret > 0) {
        sendPort.send(_ErrorMsg('KV cache full, cannot generate more tokens'));
        return;
      }
    } catch (e) {
      sendPort.send(_ErrorMsg('Decode chunk error: ${e.toString()}'));
      rethrow;
    } finally {
      bindings.llama_batch_free(batch);
    }
  }

  /// 同步处理 tokens 并更新 KV cache（适用于非流式/预热场景）
  void processTokens(List<int> tokens) {
    if (tokens.isEmpty) return;

    final nTokens = tokens.length;
    _logger.debug('Processing $nTokens tokens');

    final tokenArray = calloc<Int32>(nTokens);
    try {
      for (var i = 0; i < nTokens; i++) {
        tokenArray.elementAt(i).value = tokens[i];
      }

      final batch = bindings.llama_batch_get_one(tokenArray, nTokens);
      _logger.debug('Created batch with $nTokens tokens');

      final ret = bindings.llama_decode(handle, batch);
      if (ret < 0) {
        _logger.error('llama_decode failed with code $ret');
        throw LlamaContextInitException('Failed to decode batch: $ret');
      } else if (ret > 0) {
        _logger.warning('llama_decode returned $ret (no KV slot available)');
        throw LlamaContextInitException('KV cache full, cannot decode batch');
      }

      _nPast += nTokens;
      _kvCache?.addProcessed(nTokens);
      _logger.debug('Processed $nTokens tokens, nPast now: $_nPast');
    } catch (e) {
      _logger.error('Error processing tokens: $e');
      rethrow;
    } finally {
      calloc.free(tokenArray);
      _logger.debug('Freed token array');
    }
  }

  /// 重置上下文（清空 KV cache）
  void reset() {
    _logger.debug('Resetting context, current nPast: $_nPast');
    _nPast = 0;
    _kvCache?.reset();
    // 重置 sampler 状态
    if (_samplerChain != null) {
      bindings.llama_sampler_reset(_samplerChain!);
      _logger.debug('Reset sampler chain');
    }
    _logger.info('Context reset successfully');
  }

  /// 获取剩余上下文空间
  int get remainingContext => _config.nCtx - _nPast;

  /// 是否需要截断
  bool get needsTruncation => remainingContext < _config.nBatch;

  @override
  void dispose() {
    if (_disposed) return;

    _logger.info('Disposing context...');

    if (_samplerChain != null) {
      bindings.llama_sampler_free(_samplerChain!);
      _samplerChain = null;
    }

    if (_ctxPtr != null) {
      bindings.llama_free(_ctxPtr!);
      _ctxPtr = null;
    }

    _kvCache?.dispose();
    _kvCache = null;
    _disposed = true;

    _logger.info('Context disposed');
  }
}

// ── Isolate 通信数据结构 ─────────────────────────────────────────────────────

/// Isolate 入口参数（所有字段均为可跨 Isolate 传递的基本类型）
class _IsolateArgs {
  final SendPort sendPort;
  final int ctxAddress;
  final int samplerAddress;
  final int vocabAddress;
  final List<int> inputTokens;
  final int maxTokens;
  final int grammarAddress;
  final int nUBatch;
  final int currentNPast;

  const _IsolateArgs({
    required this.sendPort,
    required this.ctxAddress,
    required this.samplerAddress,
    required this.vocabAddress,
    required this.inputTokens,
    required this.maxTokens,
    required this.grammarAddress,
    required this.nUBatch,
    required this.currentNPast,
  });
}

/// Isolate 发回主 Isolate 的 token 消息
class _TokenMsg {
  final int token;
  final bool isEnd;
  final int nPast;

  const _TokenMsg({required this.token, required this.isEnd, required this.nPast});
}

/// Isolate 发回主 Isolate 的错误消息
class _ErrorMsg {
  final String message;
  const _ErrorMsg(this.message);
}
