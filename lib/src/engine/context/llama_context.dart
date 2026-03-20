import 'dart:ffi';
import 'dart:convert';

import 'package:ffi/ffi.dart';

import 'package:llama_native/src/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/engine/model/llama_model.dart';
import 'package:llama_native/src/engine/context/inference_config.dart';
import 'package:llama_native/src/engine/context/performance_metrics.dart';
import 'package:llama_native/src/engine/context/token_generation.dart';
import 'package:llama_native/src/engine/sampling/sampling_config.dart';
import 'package:llama_native/src/engine/cache/kv_cache_manager.dart';
import 'package:llama_native/src/log/logger.dart';

import 'package:llama_native/src/engine/exceptions/llama_exceptions.dart';

class LlamaContext {
  final LlamaModel _model;
  final InferenceConfig _config;
  final Logger _logger;

  Pointer<bindings.llama_context>? _ctxPtr;
  Pointer<bindings.llama_sampler>? _samplerChain;
  Pointer<bindings.llama_sampler>? _grammarSampler;
  KVCacheManager? _kvCache;
  int _nPast = 0;
  final List<int> _prevTokens = [];
  int _nPrev = 64;

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

    // 直接创建context参数
    final ctxParams = bindings.llama_context_default_params();
    ctxParams.n_ctx = _config.nCtx;
    ctxParams.n_batch = _config.nBatch;
    ctxParams.n_ubatch = _config.nUBatch;
    ctxParams.n_threads = _config.nThreads;
    ctxParams.n_seq_max = 1;
    ctxParams.embeddings = false;

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

    _nPrev = sampling.penaltyLastN > 0 ? sampling.penaltyLastN : 64;

    if (sampling.useMirostat) {
      _buildMirostatChain(chain, sampling);
    } else {
      _buildStandardChain(chain, sampling);
    }

    return chain;
  }

  void _buildMirostatChain(Pointer<bindings.llama_sampler> chain, SamplingConfig sampling) {
    if (sampling.hasDynatemp) {
      bindings.llama_sampler_chain_add(
        chain,
        bindings.llama_sampler_init_temp_ext(sampling.temperature, sampling.dynatempRange, sampling.dynatempExponent),
      );
    } else {
      bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_temp(sampling.temperature));
    }

    final nVocab = bindings.llama_n_vocab(_model.vocab);

    if (sampling.mirostat == 1) {
      bindings.llama_sampler_chain_add(
        chain,
        bindings.llama_sampler_init_mirostat(nVocab, sampling.seed, sampling.mirostatTau, sampling.mirostatEta, 100),
      );
    } else {
      bindings.llama_sampler_chain_add(
        chain,
        bindings.llama_sampler_init_mirostat_v2(sampling.seed, sampling.mirostatTau, sampling.mirostatEta),
      );
    }
  }

  void _buildStandardChain(Pointer<bindings.llama_sampler> chain, SamplingConfig sampling) {
    bool useAdaptiveP = false;

    for (final samplerType in sampling.samplers) {
      switch (samplerType) {
        case SamplerType.penalties:
          if (sampling.hasPenalties) {
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
          break;

        case SamplerType.dry:
          if (sampling.hasDry) {
            _addDrySampler(chain, sampling);
          }
          break;

        case SamplerType.topNSigma:
          if (sampling.hasTopNSigma) {
            bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_top_n_sigma(sampling.topNSigma));
          }
          break;

        case SamplerType.topK:
          if (sampling.topK > 0) {
            bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_top_k(sampling.topK));
          }
          break;

        case SamplerType.typicalP:
          if (sampling.hasTypicalP) {
            bindings.llama_sampler_chain_add(
              chain,
              bindings.llama_sampler_init_typical(sampling.typP, sampling.minKeep),
            );
          }
          break;

        case SamplerType.topP:
          if (sampling.hasTopP) {
            bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_top_p(sampling.topP, sampling.minKeep));
          }
          break;

        case SamplerType.minP:
          if (sampling.hasMinP) {
            bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_min_p(sampling.minP, sampling.minKeep));
          }
          break;

        case SamplerType.xtc:
          if (sampling.hasXtc) {
            bindings.llama_sampler_chain_add(
              chain,
              bindings.llama_sampler_init_xtc(
                sampling.xtcProbability,
                sampling.xtcThreshold,
                sampling.minKeep,
                sampling.seed,
              ),
            );
          }
          break;

        case SamplerType.temperature:
          if (sampling.isGreedy) {
            // Greedy will be added at the end
          } else if (sampling.hasDynatemp) {
            bindings.llama_sampler_chain_add(
              chain,
              bindings.llama_sampler_init_temp_ext(
                sampling.temperature,
                sampling.dynatempRange,
                sampling.dynatempExponent,
              ),
            );
          } else {
            bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_temp(sampling.temperature));
          }
          break;

        case SamplerType.adaptiveP:
          useAdaptiveP = true;
          break;

        default:
          break;
      }
    }

    if (sampling.isGreedy) {
      bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_greedy());
    } else if (useAdaptiveP && sampling.hasAdaptiveP) {
      bindings.llama_sampler_chain_add(
        chain,
        bindings.llama_sampler_init_adaptive_p(sampling.adaptiveTarget, sampling.adaptiveDecay, sampling.seed),
      );
    } else {
      bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_dist(sampling.seed));
    }
  }

  void _addDrySampler(Pointer<bindings.llama_sampler> chain, SamplingConfig sampling) {
    final breakers = sampling.drySequenceBreakers;
    final breakersPtr = calloc<Pointer<Char>>(breakers.length);

    try {
      for (var i = 0; i < breakers.length; i++) {
        breakersPtr[i] = breakers[i].toNativeUtf8().cast<Char>();
      }

      final nCtxTrain = bindings.llama_n_ctx_train(_model.handle);

      bindings.llama_sampler_chain_add(
        chain,
        bindings.llama_sampler_init_dry(
          _model.vocab,
          nCtxTrain,
          sampling.dryMultiplier,
          sampling.dryBase,
          sampling.dryAllowedLength,
          sampling.dryPenaltyLastN,
          breakersPtr,
          breakers.length,
        ),
      );
    } finally {
      for (var i = 0; i < breakers.length; i++) {
        calloc.free(breakersPtr[i]);
      }
      calloc.free(breakersPtr);
    }
  }

  void setGrammar(Pointer<bindings.llama_sampler>? grammarSampler) {
    if (_grammarSampler != null && _grammarSampler != nullptr) {
      bindings.llama_sampler_free(_grammarSampler!);
    }
    _grammarSampler = grammarSampler;
  }

  Pointer<bindings.llama_context> get handle {
    if (_ctxPtr == null) {
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

  LlamaModel get model => _model;

  Pointer<bindings.llama_sampler> get sampler {
    if (_samplerChain == null) {
      throw StateError('Context is disposed');
    }
    return _samplerChain!;
  }

  void decode(List<int> tokens) {
    if (_ctxPtr == null) throw StateError('Context is disposed');
    if (tokens.isEmpty) return;

    final nTokens = tokens.length;
    final batchSize = _config.nBatch;

    if (nTokens <= batchSize) {
      _decodeBatch(tokens);
    } else {
      for (var offset = 0; offset < nTokens; offset += batchSize) {
        final end = (offset + batchSize < nTokens) ? offset + batchSize : nTokens;
        final batch = tokens.sublist(offset, end);
        _decodeBatch(batch);
      }
    }
  }

  void _decodeBatch(List<int> tokens) {
    if (tokens.isEmpty) return;

    final nTokens = tokens.length;
    final startPos = _kvCache?.allocatePositions(nTokens) ?? _nPast;

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

      final ret = bindings.llama_decode(_ctxPtr!, batch);
      if (ret < 0) {
        throw LlamaException.context('llama_decode failed: $ret');
      } else if (ret > 0) {
        throw LlamaException.kvCache('KV cache full, cannot decode batch');
      }

      _prevTokens.addAll(tokens);
      if (_prevTokens.length > _nPrev) {
        final removeCount = _prevTokens.length - _nPrev;
        _prevTokens.removeRange(0, removeCount);
      }

      _nPast = startPos + nTokens;
    } finally {
      bindings.llama_batch_free(batch);
    }
  }

  int sample({bool grammarFirst = false}) {
    if (_ctxPtr == null) throw StateError('Context is disposed');

    bindings.llama_synchronize(_ctxPtr!);

    if (_grammarSampler != null && _grammarSampler != nullptr) {
      return _sampleWithGrammar(grammarFirst);
    }

    final token = bindings.llama_sampler_sample(_samplerChain!, _ctxPtr!, -1);
    _acceptToken(token);
    return token;
  }

  int _sampleWithGrammar(bool grammarFirst) {
    final token = bindings.llama_sampler_sample(_samplerChain!, _ctxPtr!, -1);

    final isValid = _checkGrammarToken(token);
    if (isValid) {
      _acceptToken(token);
      return token;
    }

    _logger.debug('Token $token rejected by grammar, resampling');
    final resampledToken = bindings.llama_sampler_sample(_grammarSampler!, _ctxPtr!, -1);
    _acceptToken(resampledToken);
    return resampledToken;
  }

  bool _checkGrammarToken(int token) {
    return true;
  }

  void _acceptToken(int token) {
    bindings.llama_sampler_accept(_samplerChain!, token);
    if (_grammarSampler != null && _grammarSampler != nullptr) {
      bindings.llama_sampler_accept(_grammarSampler!, token);
    }

    _prevTokens.add(token);
    if (_prevTokens.length > _nPrev) {
      _prevTokens.removeAt(0);
    }
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
    if (_ctxPtr == null) throw StateError('Context is disposed');

    final pos = _kvCache?.allocatePositions(1) ?? _nPast;
    final batch = bindings.llama_batch_init(1, 0, 1);
    try {
      batch.token.elementAt(0).value = token;
      batch.pos.elementAt(0).value = pos;
      batch.n_seq_id.elementAt(0).value = 1;
      batch.seq_id.elementAt(0).value.elementAt(0).value = 0;
      batch.logits.elementAt(0).value = 1;
      batch.n_tokens = 1;

      final ret = bindings.llama_decode(_ctxPtr!, batch);
      if (ret != 0) {
        throw LlamaException.inference('Decode failed: $ret');
      }

      _nPast = pos + 1;
    } finally {
      bindings.llama_batch_free(batch);
    }
  }

  void setKeepPrefixTokens(List<int> tokens) {
    _kvCache?.setKeepPrefix(tokens.length);
    _logger.info('Set keep prefix to ${tokens.length} tokens');
  }

  void reset() {
    _logger.debug('Resetting context, current nPast: $_nPast');
    _nPast = 0;
    _prevTokens.clear();
    _kvCache?.reset();
    if (_samplerChain != null) {
      bindings.llama_sampler_reset(_samplerChain!);
      _logger.debug('Reset sampler chain');
    }
    if (_grammarSampler != null && _grammarSampler != nullptr) {
      bindings.llama_sampler_reset(_grammarSampler!);
      _logger.debug('Reset grammar sampler');
    }
    _logger.info('Context reset successfully');
  }

  int get remainingContext => _kvCache?.nRemain ?? (_config.nCtx - _nPast);

  bool get needsTruncation => _kvCache?.needsTruncation ?? (remainingContext < _config.nBatch);

  int get keepPrefix => _kvCache?.keepPrefix ?? 0;

  KVCacheStatus get kvCacheStatus => _kvCache?.getStatus() ?? KVCacheStatus.empty();

  void fullResetKVCache() {
    _kvCache?.fullReset();
    _nPast = 0;
    _prevTokens.clear();
    _logger.info('KV cache fully reset');
  }

  void prepareSequentialPrefill() {
    _kvCache?.prepareSequentialPrefill();
    _nPast = 0;
    _prevTokens.clear();
    _logger.info('Prepared for sequential prefill');
  }

  PerformanceMetrics get performanceMetrics {
    if (_ctxPtr == null) {
      return PerformanceMetrics.empty();
    }

    final perfContext = bindings.llama_perf_context(_ctxPtr!);
    final perfSampler = _samplerChain != null ? bindings.llama_perf_sampler(_samplerChain!) : null;

    return PerformanceMetrics(
      tStartMs: perfContext.t_start_ms,
      tLoadMs: perfContext.t_load_ms,
      tPromptEvalMs: perfContext.t_p_eval_ms,
      tEvalMs: perfContext.t_eval_ms,
      nPromptEval: perfContext.n_p_eval,
      nEval: perfContext.n_eval,
      nReused: perfContext.n_reused,
      tSampleMs: perfSampler?.t_sample_ms ?? 0,
      nSample: perfSampler?.n_sample ?? 0,
    );
  }

  void resetPerformanceMetrics() {
    if (_ctxPtr != null && _ctxPtr != nullptr) {
      bindings.llama_perf_context_reset(_ctxPtr!);
    }
    if (_samplerChain != null && _samplerChain != nullptr) {
      bindings.llama_perf_sampler_reset(_samplerChain!);
    }
    _logger.debug('Performance metrics reset');
  }

  Pointer<bindings.llama_context> get ctxPtr {
    if (_ctxPtr == null) {
      throw StateError('Context is disposed');
    }
    return _ctxPtr!;
  }

  void dispose() {
    if (_ctxPtr == null) return;

    _logger.info('Disposing context...');

    if (_grammarSampler != null && _grammarSampler != nullptr) {
      bindings.llama_sampler_free(_grammarSampler!);
      _grammarSampler = null;
    }

    if (_samplerChain != null) {
      bindings.llama_sampler_free(_samplerChain!);
      _samplerChain = null;
    }

    bindings.llama_free(_ctxPtr!);
    _ctxPtr = null;

    _kvCache?.dispose();
    _kvCache = null;

    _logger.info('Context disposed');
  }
}
