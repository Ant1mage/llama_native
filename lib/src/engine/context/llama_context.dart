import 'dart:ffi';
import 'dart:convert';

import 'package:ffi/ffi.dart';

import 'package:llama_native/src/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/engine/backend/llama_backend.dart';
import 'package:llama_native/src/engine/backend/llama_backend_config.dart';
import 'package:llama_native/src/engine/model/llama_model.dart';
import 'package:llama_native/src/engine/context/inference_config.dart';
import 'package:llama_native/src/engine/sampling/sampling_config.dart';
import 'package:llama_native/src/engine/cache/kv_cache_manager.dart';
import 'package:llama_native/src/log/logger.dart';
import 'package:llama_native/src/utils/disposable.dart';
import 'package:llama_native/src/engine/exceptions/llama_exceptions.dart';

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
  List<int> _keepPrefixTokens = [];
  final List<int> _recentTokens = [];
  static const int _maxRecentTokens = 1024;

  String _conversationSummary = '';
  String Function(String conversationText)? _summarizeCallback;
  static const String _defaultSummaryPrompt = '请用简洁的语言总结以下对话内容，保留关键信息，不超过200字：\n\n';

  LlamaContext._(this._model, this._config) : _logger = Logger('LlamaContext');

  factory LlamaContext.create(LlamaModel model, InferenceConfig config) {
    final context = LlamaContext._(model, config);
    context._initialize();
    return context;
  }

  void setSummarizeCallback(String Function(String conversationText)? callback) {
    _summarizeCallback = callback;
  }

  String get conversationSummary => _conversationSummary;

  void setConversationSummary(String summary) {
    _conversationSummary = summary;
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
      throw LlamaException.context('llama_init_from_model returned null');
    }
    _ctxPtr = ptr;

    _samplerChain = _buildSamplerChain(_config.sampling);

    _kvCache = KVCacheManager(nCtx: _config.nCtx, ctx: _ctxPtr);

    _logger.info('Context initialized');
  }

  Pointer<bindings.llama_sampler> _buildSamplerChain(SamplingConfig sampling) {
    final chainParams = bindings.llama_sampler_chain_default_params();
    final chain = bindings.llama_sampler_chain_init(chainParams);

    if (chain == nullptr) {
      throw LlamaException.context('Failed to init sampler chain');
    }

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
      bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_greedy());
    } else {
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
      bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_dist(0xFFFFFFFF));
    }

    return chain;
  }

  Pointer<bindings.llama_context> get handle {
    if (_ctxPtr == null || _disposed) {
      throw StateError('Context is disposed');
    }
    return _ctxPtr!;
  }

  KVCacheManager get kvCache {
    if (_kvCache == null) {
      throw StateError('KVCache not initialized');
    }
    return _kvCache!;
  }

  int get nPast => _nPast;

  @override
  bool get isDisposed => _disposed;

  LlamaModel get model => _model;

  Pointer<bindings.llama_sampler> get sampler {
    if (_samplerChain == null || _disposed) {
      throw StateError('Context is disposed');
    }
    return _samplerChain!;
  }

  void decode(List<int> tokens) {
    if (_disposed) throw StateError('Context is disposed');
    if (tokens.isEmpty) return;

    final nTokens = tokens.length;
    final batchSize = _config.nBatch;

    if (nTokens <= batchSize) {
      _decodeBatch(tokens, checkOverflow: true);
    } else {
      for (var offset = 0; offset < nTokens; offset += batchSize) {
        final end = (offset + batchSize < nTokens) ? offset + batchSize : nTokens;
        final batch = tokens.sublist(offset, end);
        _decodeBatch(batch, checkOverflow: true);
      }
    }
  }

  void _decodeBatchRebuild(List<int> tokens) {
    if (tokens.isEmpty) return;

    final nTokens = tokens.length;
    final batchSize = _config.nBatch;

    if (nTokens <= batchSize) {
      _decodeBatch(tokens, checkOverflow: false);
    } else {
      for (var offset = 0; offset < nTokens; offset += batchSize) {
        final end = (offset + batchSize < nTokens) ? offset + batchSize : nTokens;
        final batch = tokens.sublist(offset, end);
        _decodeBatch(batch, checkOverflow: false);
      }
    }
  }

  void _decodeBatch(List<int> tokens, {bool checkOverflow = true}) {
    if (tokens.isEmpty) return;

    final nTokens = tokens.length;

    if (checkOverflow && _nPast + nTokens > _config.nCtx) {
      _logger.warning('KV cache would overflow: _nPast=$_nPast + nTokens=$nTokens > nCtx=${_config.nCtx}');
      _autoTruncateCache(nTokens);
    }

    final batch = bindings.llama_batch_init(nTokens, 0, 1);

    try {
      for (var i = 0; i < nTokens; i++) {
        batch.token.elementAt(i).value = tokens[i];
        batch.pos.elementAt(i).value = _nPast + i;
        batch.n_seq_id.elementAt(i).value = 1;
        batch.seq_id.elementAt(i).value.elementAt(0).value = 0;
        batch.logits.elementAt(i).value = 0;
      }
      batch.logits.elementAt(nTokens - 1).value = 1;
      batch.n_tokens = nTokens;

      final ret = bindings.llama_decode(_ctxPtr!, batch);
      if (ret < 0) {
        throw LlamaException.context('llama_decode failed: $ret');
      } else if (ret > 0) {
        throw LlamaException.kvCache('KV cache full, cannot decode batch');
      }

      _recentTokens.addAll(tokens);
      if (_recentTokens.length > _maxRecentTokens) {
        final removeCount = _recentTokens.length - _maxRecentTokens;
        _recentTokens.removeRange(0, removeCount);
      }

      _nPast += nTokens;
      _kvCache?.addProcessed(nTokens);
    } finally {
      bindings.llama_batch_free(batch);
    }
  }

  String? _pendingSummarizationText;
  int _pendingNeededTokens = 0;

  bool get needsSummarization => _pendingSummarizationText != null;

  String? getSummarizationRequest() {
    final text = _pendingSummarizationText;
    _pendingSummarizationText = null;
    return text;
  }

  void applySummaryAndRebuild(String summary) {
    if (summary.isEmpty) {
      _logger.warning('Empty summary provided, using fallback');
      _fallbackRebuild(_pendingNeededTokens);
      return;
    }

    _conversationSummary = summary;
    _logger.info('Applying summary: ${summary.length} chars');

    final mem = bindings.llama_get_memory(_ctxPtr!);
    bindings.llama_memory_clear(mem, true);
    bindings.llama_synchronize(_ctxPtr!);
    _nPast = 0;

    final tokensToRestore = <int>[];

    if (_keepPrefixTokens.isNotEmpty) {
      tokensToRestore.addAll(_keepPrefixTokens);
    }

    if (_conversationSummary.isNotEmpty) {
      final summaryPrompt = '对话历史摘要：$_conversationSummary\n\n';
      final summaryTokens = _model.tokenize(summaryPrompt, addBos: false);
      tokensToRestore.addAll(summaryTokens);
      _logger.info('Added summary tokens: ${summaryTokens.length}');
    }

    final availableSpace = _config.nCtx - tokensToRestore.length - _pendingNeededTokens - (_config.nCtx ~/ 8);
    if (availableSpace > 0 && _recentTokens.isNotEmpty) {
      final recentCount = availableSpace < _recentTokens.length ? availableSpace : _recentTokens.length;
      final recentStart = _recentTokens.length - recentCount;
      tokensToRestore.addAll(_recentTokens.sublist(recentStart));
      _logger.info('Added recent tokens: $recentCount');
    }

    if (tokensToRestore.isNotEmpty) {
      _logger.info('Re-decoding ${tokensToRestore.length} tokens with summary');
      _decodeBatchRebuild(tokensToRestore);
    }

    _pendingNeededTokens = 0;
  }

  void _fallbackRebuild(int neededTokens) {
    final mem = bindings.llama_get_memory(_ctxPtr!);
    bindings.llama_memory_clear(mem, true);
    bindings.llama_synchronize(_ctxPtr!);
    _nPast = 0;

    final tokensToRestore = <int>[];

    if (_keepPrefixTokens.isNotEmpty) {
      tokensToRestore.addAll(_keepPrefixTokens);
      _logger.info('Will restore ${_keepPrefixTokens.length} keep prefix tokens');
    }

    final availableSpace = _config.nCtx - tokensToRestore.length - neededTokens - (_config.nCtx ~/ 8);
    if (availableSpace > 0 && _recentTokens.isNotEmpty) {
      final recentCount = availableSpace < _recentTokens.length ? availableSpace : _recentTokens.length;
      final recentStart = _recentTokens.length - recentCount;
      tokensToRestore.addAll(_recentTokens.sublist(recentStart));
      _logger.info('Will restore $recentCount recent tokens');
    }

    if (tokensToRestore.isNotEmpty) {
      _logger.info('Re-decoding ${tokensToRestore.length} tokens after cache clear');
      _decodeBatchRebuild(tokensToRestore);
    }
  }

  void _autoTruncateCache(int neededTokens) {
    _logger.warning('Context full, preparing for cache rebuild');

    if (_summarizeCallback != null && _recentTokens.isNotEmpty) {
      final conversationText = _model.detokenize(_recentTokens);
      _pendingSummarizationText = conversationText;
      _pendingNeededTokens = neededTokens;
      _logger.info('Requesting summarization for ${conversationText.length} chars');
      return;
    }

    _fallbackRebuild(neededTokens);
  }

  int sample({Pointer<bindings.llama_sampler>? grammarSampler}) {
    if (_disposed) throw StateError('Context is disposed');

    int token;
    if (grammarSampler != null && grammarSampler != nullptr) {
      token = bindings.llama_sampler_sample(grammarSampler, _ctxPtr!, -1);
      bindings.llama_sampler_accept(grammarSampler, token);
    } else {
      token = bindings.llama_sampler_sample(_samplerChain!, _ctxPtr!, -1);
      bindings.llama_sampler_accept(_samplerChain!, token);
    }

    return token;
  }

  bool isEos(int token) {
    return bindings.llama_token_is_eog(_model.vocab, token);
  }

  String detokenizeOne(int token) {
    var bufferSize = bindings.llama_token_to_piece(_model.vocab, token, nullptr, 0, 0, true);

    if (bufferSize < 0) {
      bufferSize = -bufferSize;
    }

    if (bufferSize == 0) {
      return '';
    }

    final pieceBuffer = calloc<Char>(bufferSize);
    try {
      final actualSize = bindings.llama_token_to_piece(_model.vocab, token, pieceBuffer, bufferSize, 0, true);

      if (actualSize > 0) {
        final allBytes = <int>[];
        for (var i = 0; i < actualSize; i++) {
          final byte = pieceBuffer.elementAt(i).value;
          allBytes.add(byte < 0 ? byte + 256 : byte);
        }
        try {
          return utf8.decode(allBytes, allowMalformed: true);
        } catch (e) {
          return String.fromCharCodes(allBytes);
        }
      }
      return '';
    } finally {
      calloc.free(pieceBuffer);
    }
  }

  void decodeOne(int token) {
    if (_disposed) throw StateError('Context is disposed');

    final batch = bindings.llama_batch_init(1, 0, 1);
    try {
      batch.token.elementAt(0).value = token;
      batch.pos.elementAt(0).value = _nPast;
      batch.n_seq_id.elementAt(0).value = 1;
      batch.seq_id.elementAt(0).value.elementAt(0).value = 0;
      batch.logits.elementAt(0).value = 1;
      batch.n_tokens = 1;

      final ret = bindings.llama_decode(_ctxPtr!, batch);
      if (ret != 0) {
        throw LlamaException.inference('Decode failed: $ret');
      }

      _nPast += 1;
      _kvCache?.addProcessed(1);
    } finally {
      bindings.llama_batch_free(batch);
    }
  }

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
        throw LlamaException.inference('Failed to decode batch: $ret');
      } else if (ret > 0) {
        _logger.warning('llama_decode returned $ret (no KV slot available)');
        throw LlamaException.kvCache('KV cache full, cannot decode batch');
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

  void setKeepPrefixTokens(List<int> tokens) {
    _keepPrefixTokens = List.from(tokens);
    _kvCache?.setKeepPrefix(tokens.length);
    _logger.info('Set keep prefix to ${tokens.length} tokens');
  }

  void reset() {
    _logger.debug('Resetting context, current nPast: $_nPast');
    _nPast = 0;
    _recentTokens.clear();
    _kvCache?.reset();
    if (_samplerChain != null) {
      bindings.llama_sampler_reset(_samplerChain!);
      _logger.debug('Reset sampler chain');
    }
    _logger.info('Context reset successfully');
  }

  int get remainingContext => _config.nCtx - _nPast;

  bool get needsTruncation => remainingContext < _config.nBatch;

  int get keepPrefix => _kvCache?.keepPrefix ?? 0;

  Pointer<bindings.llama_context> get ctxPtr {
    if (_ctxPtr == null || _disposed) {
      throw StateError('Context is disposed');
    }
    return _ctxPtr!;
  }

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
