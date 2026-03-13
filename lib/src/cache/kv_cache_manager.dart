import 'package:llama_native/src/logging/logger.dart';
import 'package:llama_native/src/utils/disposable.dart';

/// Llama KV Cache 管理器 (简化高级封装)
class KVCacheManager with Disposable {
  final Logger _logger = Logger('LlamaKVCacheManager');
  final int _nCtx;
  int _nPast = 0;
  int _keepPrefix = 0;
  int? _windowSize;
  bool _disposed = false;

  /// 创建 KV Cache 管理器
  KVCacheManager({required int nCtx, int? windowSize}) : _nCtx = nCtx {
    if (windowSize != null) {
      _windowSize = windowSize;
    }
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

    if (_keepPrefix > 0) {
      final historyLength = targetLength - _keepPrefix;
      if (historyLength > 0) {
        _nPast = _keepPrefix + historyLength;
      }
    } else {
      _nPast = targetLength;
    }

    _logger.info('Auto-truncated to $_nPast tokens');
  }

  /// 应用滑动窗口
  void _applySlidingWindow() {
    if (_windowSize == null) return;

    final targetLength = _windowSize!;

    if (_keepPrefix > 0) {
      final historyLength = targetLength - _keepPrefix;
      if (historyLength > 0) {
        _nPast = _keepPrefix + historyLength;
      }
    } else {
      _nPast = targetLength;
    }

    _logger.debug('Applied sliding window: $_nPast tokens');
  }

  /// 重置 KV Cache
  void reset() {
    _nPast = 0;
    _logger.debug('KV Cache cleared');
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
