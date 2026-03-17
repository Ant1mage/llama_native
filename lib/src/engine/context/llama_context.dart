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

      _nPast += nTokens;
      _kvCache?.addProcessed(nTokens);
    } finally {
      bindings.llama_batch_free(batch);
    }
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

  void reset() {
    _logger.debug('Resetting context, current nPast: $_nPast');
    _nPast = 0;
    _kvCache?.reset();
    if (_samplerChain != null) {
      bindings.llama_sampler_reset(_samplerChain!);
      _logger.debug('Reset sampler chain');
    }
    _logger.info('Context reset successfully');
  }

  int get remainingContext => _config.nCtx - _nPast;

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
