import 'dart:async';
import 'dart:io';
import 'package:llama_native/src/context/inference_config.dart';
import 'package:llama_native/src/context/token_generation.dart';

import 'package:llama_native/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/backend/llama_backend.dart';
import 'package:llama_native/src/model/llama_model.dart';
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

    final tokens = List<int>.from(inputTokens);

    for (var i = 0; i < maxTokens; i++) {
      // 生成下一个 token
      final result = _generateNextSync(tokens);

      yield result;

      if (result.isEnd) {
        break;
      }

      // 使用生成的 token 继续
      tokens.clear();
      tokens.add(result.token);
    }
  }

  /// 同步生成单个 token
  TokenGeneration _generateNextSync(List<int> tokens) {
    if (_disposed) throw StateError('Context is disposed');

    final lastToken = tokens.last;
    final text = _model.detokenize([lastToken]);
    final eosToken = bindings.llama_token_eos(_model.vocab);

    return TokenGeneration(token: lastToken, text: text, isEnd: lastToken == eosToken);
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
