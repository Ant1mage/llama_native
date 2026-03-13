import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

import 'package:llama_native/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/backend/llama_backend.dart';
import 'package:llama_native/src/model/llama_model.dart';
import 'package:llama_native/src/context/inference_config.dart';
import 'package:llama_native/src/context/token_generation.dart';
import 'package:llama_native/src/sampling/sampling_config.dart';
import 'package:llama_native/src/cache/kv_cache_manager.dart';
import 'package:llama_native/src/logging/logger.dart';
import 'package:llama_native/src/utils/disposable.dart';
import 'package:llama_native/src/exceptions/llama_exceptions.dart';

/// 推理执行引擎
class LlamaContext with Disposable {
  final LlamaModel _model;
  final InferenceConfig _config;
  final Logger _logger;

  dynamic _ctxPtr;
  KVCacheManager? _kvCache;
  bool _disposed = false;
  int _nPast = 0;

  /// 私有构造函数
  LlamaContext._(this._model, this._config) : _logger = Logger('LlamaContext');

  /// 创建上下文（同步）
  factory LlamaContext.create(LlamaModel model, InferenceConfig config) {
    final context = LlamaContext._(model, config);
    context._initialize();
    return context;
  }

  /// 初始化上下文
  void _initialize() {
    if (_ctxPtr != null) {
      throw StateError('Context already initialized');
    }

    _logger.info('Creating context: n_ctx=${_config.nCtx}, n_batch=${_config.nBatch}');

    final backend = LlamaBackend.instance;
    final ctxParams = backend.getContextParams(nCtx: _config.nCtx, nBatch: _config.nBatch, nThreads: _config.nThreads);

    // 创建上下文
    final ptr = bindings.llama_init_from_model(_model.handle, ctxParams);

    _ctxPtr = ptr;

    // 初始化 KV Cache 管理器
    _kvCache = KVCacheManager(nCtx: _config.nCtx);

    _logger.info('Context initialized');
  }

  /// 获取上下文指针
  dynamic get handle {
    if (_ctxPtr == null || _disposed) {
      throw StateError('Context is disposed');
    }
    return _ctxPtr;
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

  /// 流式生成（同步）
  Stream<TokenGeneration> generateStream(List<int> inputTokens, {int maxTokens = 256}) async* {
    if (_disposed) throw StateError('Context is disposed');

    // 首先处理输入 tokens
    _processTokens(inputTokens);

    for (var i = 0; i < maxTokens; i++) {
      // 生成下一个 token
      final result = _generateNextSync();

      yield result;

      if (result.isEnd) {
        break;
      }

      // 使用生成的 token 继续
      _processTokens([result.token]);
    }
  }

  /// 处理 tokens 并更新 KV cache
  void _processTokens(List<int> tokens) {
    if (tokens.isEmpty) return;

    // 分配并初始化 token 数组
    final tokenArray = calloc<Int32>(tokens.length);
    for (var i = 0; i < tokens.length; i++) {
      tokenArray[i] = tokens[i];
    }

    try {
      // 使用 llama_batch_get_one 获取 batch
      final batch = bindings.llama_batch_get_one(tokenArray, tokens.length);

      // 设置位置
      for (var i = 0; i < tokens.length; i++) {
        batch.pos.elementAt(i).value = _nPast + i;
      }

      // 解码批次
      final result = bindings.llama_decode(handle, batch);
      if (result < 0) {
        _logger.error('llama_decode failed with code $result');
        throw LlamaContextInitException('Failed to decode batch');
      }

      _nPast += tokens.length;
      _kvCache?.addProcessed(tokens.length);

      // 释放 batch
      bindings.llama_batch_free(batch);
    } finally {
      calloc.free(tokenArray);
    }
  }

  /// 同步生成单个 token
  TokenGeneration _generateNextSync() {
    if (_disposed) throw StateError('Context is disposed');

    final vocabHandle = _model.vocab;
    const eosTokenId = 2; // 默认 EOS token ID，实际应该从 vocab 获取

    // 获取 logits (最后一个 token 的)
    final logits = bindings.llama_get_logits_ith(handle, _nPast - 1);

    if (logits == nullptr) {
      throw LlamaContextInitException('Failed to get logits');
    }

    // 应用采样
    final sampledToken = _sampleFromLogits(logits, vocabHandle);

    // detokenize
    final text = _model.detokenize([sampledToken]);

    return TokenGeneration(token: sampledToken, text: text, isEnd: sampledToken == eosTokenId);
  }

  /// 从 logits 采样 token (简化实现 - greedy sampling)
  int _sampleFromLogits(Pointer<Float> logits, Pointer<bindings.llama_vocab> vocab) {
    final vocabSize = bindings.llama_n_vocab(vocab);

    double maxLogit = double.negativeInfinity;
    int maxIndex = 0;

    // 简单的 greedy sampling (选择最大概率的 token)
    for (var i = 0; i < vocabSize; i++) {
      final logit = logits.elementAt(i).value;
      if (logit > maxLogit) {
        maxLogit = logit;
        maxIndex = i;
      }
    }

    return maxIndex;
  }

  /// 重置上下文
  void reset() {
    _nPast = 0;
    _kvCache?.reset();
    _logger.debug('Context reset');
  }

  /// 获取剩余上下文空间
  int get remainingContext => _config.nCtx - _nPast;

  /// 是否需要截断
  bool get needsTruncation => remainingContext < _config.nBatch;

  @override
  void dispose() {
    if (_disposed) return;

    _logger.info('Disposing context...');

    if (_ctxPtr != null) {
      bindings.llama_free(_ctxPtr);
      _ctxPtr = null;
    }

    _kvCache?.dispose();
    _kvCache = null;
    _disposed = true;

    _logger.info('Context disposed');
  }
}
