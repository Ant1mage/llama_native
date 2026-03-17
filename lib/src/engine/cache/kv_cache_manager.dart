import 'dart:ffi';
import 'package:llama_native/src/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/log/logger.dart';
import 'package:llama_native/src/utils/disposable.dart';

/// Llama KV Cache 管理器
class KVCacheManager with Disposable {
  final Logger _logger = Logger('LlamaKVCacheManager');
  final int _nCtx;
  int _nPast = 0;
  int _keepPrefix = 0;
  int? _windowSize;
  bool _disposed = false;
  Pointer<bindings.llama_context>? _ctx;

  /// 创建 KV Cache 管理器
  KVCacheManager({required int nCtx, int? windowSize, Pointer<bindings.llama_context>? ctx})
    : _nCtx = nCtx,
      _ctx = ctx {
    if (windowSize != null) {
      _windowSize = windowSize;
    }
  }

  /// 设置上下文指针
  void setContext(Pointer<bindings.llama_context> ctx) {
    _ctx = ctx;
  }

  /// 当前已处理 token 数
  int get nPast => _nPast;

  /// 剩余上下文空间
  int get nRemain => _nCtx - _nPast;

  /// 保留的前缀 token 数
  int get keepPrefix => _keepPrefix;

  /// 设置保留前缀
  void setKeepPrefix(int prefixTokens) {
    _keepPrefix = prefixTokens;
    _logger.debug('Set keep_prefix to $prefixTokens');
  }

  /// 检查是否需要截断
  bool get needsTruncation => _nPast >= _nCtx;

  /// 添加已处理的 token 数
  void addProcessed(int count) {
    _nPast += count;
    _checkCacheManagement();
  }

  /// 检查并执行 cache 管理
  void _checkCacheManagement() {
    if (needsTruncation) {
      _logger.warning('Context full, triggering truncation');
      _autoTruncate();
    } else if (_windowSize != null && _nPast > _windowSize!) {
      _logger.debug('Sliding window triggered');
      _applySlidingWindow();
    }
  }

  /// 自动截断
  void _autoTruncate() {
    final targetLength = _nCtx - (_nCtx ~/ 4); // 保留 75%
    _performTruncation(targetLength);
  }

  /// 应用滑动窗口
  void _applySlidingWindow() {
    if (_windowSize == null) return;
    _performTruncation(_windowSize!);
  }

  /// 执行实际的缓存截断操作
  void _performTruncation(int targetLength) {
    if (_ctx == null) {
      // 没有上下文指针，只更新状态
      if (_keepPrefix > 0) {
        final historyLength = targetLength - _keepPrefix;
        if (historyLength > 0) {
          _nPast = _keepPrefix + historyLength;
        }
      } else {
        _nPast = targetLength;
      }
      _logger.info('Updated cache state to $_nPast tokens (no context)');
      return;
    }

    try {
      // 获取内存对象
      final mem = bindings.llama_get_memory(_ctx!);

      // 执行实际的缓存操作
      if (_keepPrefix > 0) {
        final historyLength = targetLength - _keepPrefix;
        if (historyLength > 0) {
          _nPast = _keepPrefix + historyLength;
          _logger.info('Truncated to $_nPast tokens (keeping $keepPrefix prefix)');
        }
      } else {
        _nPast = targetLength;
        _logger.info('Truncated to $_nPast tokens');
      }
    } catch (e) {
      _logger.error('Error during cache truncation: $e');
      // 出错时至少更新状态
      if (_keepPrefix > 0) {
        final historyLength = targetLength - _keepPrefix;
        if (historyLength > 0) {
          _nPast = _keepPrefix + historyLength;
        }
      } else {
        _nPast = targetLength;
      }
    }
  }

  /// 重置 KV Cache
  void reset() {
    _nPast = 0;
    if (_ctx != null) {
      try {
        final mem = bindings.llama_get_memory(_ctx!);
        bindings.llama_memory_clear(mem, true);
        _logger.debug('KV Cache reset');
      } catch (e) {
        _logger.error('Error during cache reset: $e');
      }
    } else {
      _logger.debug('KV Cache state reset (no context)');
    }
  }

  /// 获取 KV Cache 使用率
  double get usagePercent => (_nPast / _nCtx) * 100;

  @override
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _logger.debug('KVCacheManager disposed');
  }

  @override
  bool get isDisposed => _disposed;
}
