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

class LlamaContext with Disposable {
  final LlamaModel _model;
  final InferenceConfig _config;
  final Logger _logger;

  Pointer<bindings.llama_context>? _ctxPtr;
  Pointer<bindings.llama_sampler>? _samplerChain;
  Pointer<bindings.llama_sampler>? _grammarSampler;
  KVCacheManager? _kvCache;
  bool _disposed = false;
  int _nPast = 0;
  List<int> _keepPrefixTokens = [];
  final List<int> _prevTokens = [];
  int _nPrev = 64;

  String _conversationSummary = '';
  String Function(String conversationText)? _summarizeCallback;

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

      _prevTokens.addAll(tokens);
      if (_prevTokens.length > _nPrev) {
        final removeCount = _prevTokens.length - _nPrev;
        _prevTokens.removeRange(0, removeCount);
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
    _prevTokens.clear();

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
    if (availableSpace > 0 && _prevTokens.isNotEmpty) {
      final recentCount = availableSpace < _prevTokens.length ? availableSpace : _prevTokens.length;
      final recentStart = _prevTokens.length - recentCount;
      tokensToRestore.addAll(_prevTokens.sublist(recentStart));
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
    _prevTokens.clear();

    final tokensToRestore = <int>[];

    if (_keepPrefixTokens.isNotEmpty) {
      tokensToRestore.addAll(_keepPrefixTokens);
      _logger.info('Will restore ${_keepPrefixTokens.length} keep prefix tokens');
    }

    final availableSpace = _config.nCtx - tokensToRestore.length - neededTokens - (_config.nCtx ~/ 8);
    if (availableSpace > 0 && _prevTokens.isNotEmpty) {
      final recentCount = availableSpace < _prevTokens.length ? availableSpace : _prevTokens.length;
      final recentStart = _prevTokens.length - recentCount;
      tokensToRestore.addAll(_prevTokens.sublist(recentStart));
      _logger.info('Will restore $recentCount recent tokens');
    }

    if (tokensToRestore.isNotEmpty) {
      _logger.info('Re-decoding ${tokensToRestore.length} tokens after cache clear');
      _decodeBatchRebuild(tokensToRestore);
    }
  }

  void _autoTruncateCache(int neededTokens) {
    _logger.warning('Context full, preparing for cache rebuild');

    if (_summarizeCallback != null && _prevTokens.isNotEmpty) {
      final conversationText = _model.detokenize(_prevTokens);
      _pendingSummarizationText = conversationText;
      _pendingNeededTokens = neededTokens;
      _logger.info('Requesting summarization for ${conversationText.length} chars');
      return;
    }

    _fallbackRebuild(neededTokens);
  }

  int sample({bool grammarFirst = false}) {
    if (_disposed) throw StateError('Context is disposed');

    bindings.llama_synchronize(_ctxPtr!);

    if (_grammarSampler != null && _grammarSampler != nullptr) {
      return _sampleWithGrammar(grammarFirst);
    }

    final token = bindings.llama_sampler_sample(_samplerChain!, _ctxPtr!, -1);
    _acceptToken(token);
    return token;
  }

  int _sampleWithGrammar(bool grammarFirst) {
    if (grammarFirst) {
      final token = bindings.llama_sampler_sample(_grammarSampler!, _ctxPtr!, -1);
      bindings.llama_sampler_apply(_samplerChain!, _getCurP());
      final selected = _getCurPSelected();
      if (selected >= 0) {
        _acceptToken(token);
        return token;
      }
    }

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

  dynamic _getCurP() {
    return nullptr;
  }

  int _getCurPSelected() {
    return -1;
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

    if (_grammarSampler != null && _grammarSampler != nullptr) {
      bindings.llama_sampler_free(_grammarSampler!);
      _grammarSampler = null;
    }

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
