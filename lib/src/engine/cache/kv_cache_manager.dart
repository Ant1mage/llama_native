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

  KVCacheManager({required int nCtx, int? windowSize, Pointer<bindings.llama_context>? ctx})
    : _nCtx = nCtx,
      _ctx = ctx {
    if (windowSize != null) {
      _windowSize = windowSize;
    }
  }

  void setContext(Pointer<bindings.llama_context> ctx) {
    _ctx = ctx;
  }

  int get nPast => _nPast;
  int get nRemain => _nCtx - _nPast;
  int get keepPrefix => _keepPrefix;

  void setKeepPrefix(int prefixTokens) {
    _keepPrefix = prefixTokens;
    _logger.debug('Set keep_prefix to $prefixTokens');
  }

  bool get needsTruncation => _nPast >= _nCtx;

  void addProcessed(int count) {
    _nPast += count;
    _checkCacheManagement();
  }

  void _checkCacheManagement() {
    if (needsTruncation) {
      _logger.warning('Context full, triggering truncation');
      _autoTruncate();
    } else if (_windowSize != null && _nPast > _windowSize!) {
      _logger.debug('Sliding window triggered');
      _applySlidingWindow();
    }
  }

  void _autoTruncate() {
    final targetLength = _nCtx - (_nCtx ~/ 4);
    _performTruncation(targetLength);
  }

  void _applySlidingWindow() {
    if (_windowSize == null) return;
    _performTruncation(_windowSize!);
  }

  void _performTruncation(int targetLength) {
    if (_ctx == null) {
      _updateStateOnly(targetLength);
      _logger.info('Updated cache state to $_nPast tokens (no context)');
      return;
    }

    try {
      final mem = bindings.llama_get_memory(_ctx!);

      final removeCount = _nPast - targetLength;
      if (removeCount <= 0) return;

      final removeStart = _keepPrefix;
      final removeEnd = removeStart + removeCount;

      final removed = bindings.llama_memory_seq_rm(mem, 0, removeStart, removeEnd);

      if (removed) {
        bindings.llama_memory_seq_add(mem, 0, removeEnd, _nPast, -removeCount);
        _nPast -= removeCount;
        _logger.info('Truncated cache: removed $removeCount tokens, nPast=$_nPast');
      } else {
        _logger.warning('Failed to remove cache entries, falling back to state update');
        _updateStateOnly(targetLength);
      }
    } catch (e) {
      _logger.error('Error during cache truncation: $e');
      _updateStateOnly(targetLength);
    }
  }

  void _updateStateOnly(int targetLength) {
    if (_keepPrefix > 0) {
      final historyLength = targetLength - _keepPrefix;
      if (historyLength > 0) {
        _nPast = _keepPrefix + historyLength;
      }
    } else {
      _nPast = targetLength;
    }
  }

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
