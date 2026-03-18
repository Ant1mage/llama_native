import 'dart:ffi';
import 'package:llama_native/src/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/log/logger.dart';
import 'package:llama_native/src/utils/disposable.dart';
import 'package:llama_native/src/engine/exceptions/llama_exceptions.dart';
import 'package:llama_native/src/engine/context/token_generation.dart';

typedef KVCacheRebuildCallback = List<int> Function(int neededTokens);

class KVCacheManager with Disposable {
  final Logger _logger = Logger('LlamaKVCacheManager');
  final int _nCtx;
  int _logicalPos = 0;
  int _keepPrefix = 0;
  int _nRecent = 256;
  int? _windowSize;
  bool _disposed = false;
  Pointer<bindings.llama_context>? _ctx;
  KVCacheRebuildCallback? _rebuildCallback;

  static const double _moeBufferPercent = 0.20;
  static const double _truncationThreshold = 0.75;

  int get _moeBufferSlots => (_nCtx * _moeBufferPercent).toInt();
  int get _safeCapacity => _nCtx - _moeBufferSlots;

  int get physicalUsedCells {
    if (_ctx == null) return 0;
    final mem = bindings.llama_get_memory(_ctx!);
    final maxPos = bindings.llama_memory_seq_pos_max(mem, 0);
    return maxPos >= 0 ? maxPos + 1 : 0;
  }

  int get logicalPosition => _logicalPos;

  int get _usedCells => physicalUsedCells;

  double get usagePercent => (_usedCells / _nCtx) * 100;
  bool get isOverSafeThreshold => _usedCells >= _safeCapacity;
  bool get isFull => _usedCells >= _nCtx;
  bool get needsTruncation => _usedCells >= (_nCtx * _truncationThreshold);

  KVCacheManager({
    required int nCtx,
    int? windowSize,
    Pointer<bindings.llama_context>? ctx,
    int nRecent = 256,
    KVCacheRebuildCallback? rebuildCallback,
  }) : _nCtx = nCtx,
       _ctx = ctx,
       _nRecent = nRecent,
       _rebuildCallback = rebuildCallback {
    if (windowSize != null) {
      _windowSize = windowSize;
    }
  }

  void setContext(Pointer<bindings.llama_context> ctx) {
    _ctx = ctx;
  }

  void setRebuildCallback(KVCacheRebuildCallback? callback) {
    _rebuildCallback = callback;
  }

  int get nPast => _logicalPos;
  int get nRemain => _nCtx - _usedCells;
  int get keepPrefix => _keepPrefix;
  int get logicalPos => _logicalPos;

  void setKeepPrefix(int prefixTokens) {
    _keepPrefix = prefixTokens;
    _logger.debug('Set keep_prefix to $prefixTokens');
  }

  void addProcessed(int count) {
    _logicalPos += count;
    _checkCacheManagement();
  }

  int allocatePositions(int count) {
    final startPos = _logicalPos;
    _logicalPos += count;
    _checkCacheManagement();
    return startPos;
  }

  void _checkCacheManagement() {
    if (isFull) {
      _logger.warning('KV cache FULL: $_usedCells/$_nCtx (${usagePercent.toStringAsFixed(1)}%)');
      _handleFullCache();
    } else if (isOverSafeThreshold) {
      _logger.warning('KV cache over safe threshold: $_usedCells/$_nCtx (${usagePercent.toStringAsFixed(1)}%)');
      _handleOverThreshold();
    } else if (needsTruncation) {
      _logger.info('KV cache near threshold: $_usedCells/$_nCtx (${usagePercent.toStringAsFixed(1)}%)');
      _autoTruncate();
    } else if (_windowSize != null && _usedCells > _windowSize!) {
      _logger.debug('Sliding window triggered');
      _applySlidingWindow();
    }
  }

  void _handleFullCache() {
    _logger.error('KV cache exhausted, cannot continue inference');
    throw LlamaException.kvCache('KV cache full, inference terminated');
  }

  void _handleOverThreshold() {
    if (_rebuildCallback != null) {
      _logger.info('Requesting cache rebuild from callback');
      try {
        final tokensToRestore = _rebuildCallback!(_nCtx ~/ 4);
        if (tokensToRestore.isNotEmpty) {
          requestRebuild(tokensToRestore);
          return;
        }
      } catch (e) {
        _logger.error('Rebuild callback failed: $e');
      }
    }

    _logger.warning('Performing emergency truncation');
    _emergencyTruncate();
  }

  void requestRebuild(List<int> tokens) {
    if (_ctx == null) {
      _logger.warning('Cannot rebuild: no context');
      return;
    }

    try {
      final mem = bindings.llama_get_memory(_ctx!);
      bindings.llama_memory_clear(mem, true);
      bindings.llama_synchronize(_ctx!);
      _logicalPos = 0;
      _logger.info('KV cache cleared for rebuild, ready for ${tokens.length} tokens');
    } catch (e) {
      _logger.error('Error clearing cache for rebuild: $e');
    }
  }

  void _emergencyTruncate() {
    if (_ctx == null) {
      _logger.warning('No context for truncation');
      return;
    }

    try {
      final mem = bindings.llama_get_memory(_ctx!);
      final usedCells = _usedCells;

      final removeStart = _keepPrefix;
      final removeEnd = usedCells - _nRecent;

      if (removeEnd <= removeStart) {
        _logger.warning('Cannot truncate: keep_prefix=$_keepPrefix, n_recent=$_nRecent, used=$usedCells');
        _logger.warning('Falling back to full clear');
        clear();
        return;
      }

      final removed = bindings.llama_memory_seq_rm(mem, 0, removeStart, removeEnd);

      if (removed) {
        bindings.llama_synchronize(_ctx!);
        final newUsed = _usedCells;
        _logger.info('Emergency truncation: removed positions $removeStart-$removeEnd');
        _logger.info('Used cells: $usedCells -> $newUsed (${(newUsed / _nCtx * 100).toStringAsFixed(1)}%)');

        if (newUsed > _safeCapacity) {
          _logger.error('Still over safe threshold after truncation, requesting full reset');
          throw LlamaException.kvCache('KV cache cannot be recovered, full reset required');
        }
      } else {
        _logger.warning('Emergency truncation failed, clearing cache');
        clear();
      }
    } catch (e) {
      if (e is LlamaException) rethrow;
      _logger.error('Error during emergency truncation: $e');
      clear();
    }
  }

  void _autoTruncate() {
    if (_ctx == null) return;

    try {
      final mem = bindings.llama_get_memory(_ctx!);
      final usedCells = _usedCells;
      final targetCells = (_nCtx * 0.6).toInt();
      final removeCount = usedCells - targetCells;

      if (removeCount <= 0) return;

      final removeStart = _keepPrefix;
      final removeEnd = removeStart + removeCount;

      if (removeEnd > usedCells) {
        _logger.warning('Invalid truncation range: $removeStart-$removeEnd (used=$usedCells)');
        return;
      }

      final removed = bindings.llama_memory_seq_rm(mem, 0, removeStart, removeEnd);

      if (removed) {
        bindings.llama_synchronize(_ctx!);
        final newUsed = _usedCells;
        _logger.info('Auto truncation: removed $removeCount cells, usage: ${usedCells} -> $newUsed');
      } else {
        _logger.warning('Auto truncation failed');
      }
    } catch (e) {
      _logger.error('Error during auto truncation: $e');
    }
  }

  void _applySlidingWindow() {
    if (_windowSize == null || _ctx == null) return;

    try {
      final mem = bindings.llama_get_memory(_ctx!);
      final usedCells = _usedCells;

      if (usedCells <= _windowSize!) return;

      final removeCount = usedCells - _windowSize!;
      final removeStart = _keepPrefix;
      final removeEnd = removeStart + removeCount;

      if (removeEnd <= removeStart) return;

      final removed = bindings.llama_memory_seq_rm(mem, 0, removeStart, removeEnd);

      if (removed) {
        bindings.llama_synchronize(_ctx!);
        _logger.info('Sliding window: removed $removeCount cells');
      }
    } catch (e) {
      _logger.error('Error during sliding window: $e');
    }
  }

  void clear() {
    if (_ctx != null) {
      try {
        final mem = bindings.llama_get_memory(_ctx!);
        bindings.llama_memory_clear(mem, true);
        bindings.llama_synchronize(_ctx!);
        _logger.info('KV Cache fully cleared');
      } catch (e) {
        _logger.error('Error during cache clear: $e');
      }
    }
    _logicalPos = 0;
  }

  void reset() {
    clear();
  }

  void syncFromContext() {
    if (_ctx == null) return;

    try {
      final mem = bindings.llama_get_memory(_ctx!);
      final actualPos = bindings.llama_memory_seq_pos_max(mem, 0);
      if (actualPos >= 0) {
        _logicalPos = actualPos + 1;
        _logger.debug('Synced logicalPos from context: $_logicalPos');
      }
    } catch (e) {
      _logger.warning('Failed to sync from context: $e');
    }
  }

  Map<String, dynamic> getStats() {
    return {
      'nCtx': _nCtx,
      'usedCells': _usedCells,
      'logicalPos': _logicalPos,
      'usagePercent': usagePercent.toStringAsFixed(1),
      'safeCapacity': _safeCapacity,
      'moeBufferSlots': _moeBufferSlots,
      'isOverSafeThreshold': isOverSafeThreshold,
      'keepPrefix': _keepPrefix,
      'nRecent': _nRecent,
    };
  }

  KVCacheStatus getStatus() {
    return KVCacheStatus(
      used: _usedCells,
      total: _nCtx,
      safeCapacity: _safeCapacity,
      moeBufferSlots: _moeBufferSlots,
      usagePercent: usagePercent,
      isOverSafeThreshold: isOverSafeThreshold,
      isFull: isFull,
    );
  }

  bool canAllocate(int tokens) {
    return _usedCells + tokens < _safeCapacity;
  }

  void checkMoEBuffer(int tokens) {
    final projectedUsage = _usedCells + tokens;
    if (projectedUsage > _safeCapacity) {
      _logger.error('MoE buffer overflow: projected=$projectedUsage, safeCapacity=$_safeCapacity');
      throw LlamaException.kvCache('MoE buffer overflow, need reset before continuing');
    }
  }

  void fullReset() {
    if (_ctx == null) {
      _logger.warning('No context for full reset');
      _logicalPos = 0;
      return;
    }

    _logger.info('Performing full KV cache reset');

    try {
      final mem = bindings.llama_get_memory(_ctx!);

      bindings.llama_memory_seq_rm(mem, -1, 0, -1);

      bindings.llama_synchronize(_ctx!);

      _logicalPos = 0;

      _logger.info('KV cache fully reset: used=$_usedCells, logicalPos=$_logicalPos');
    } catch (e) {
      _logger.error('Error during full reset: $e');
      _logicalPos = 0;
    }
  }

  void prepareSequentialPrefill() {
    if (_ctx == null) {
      _logger.warning('No context for sequential prefill');
      return;
    }

    _logger.info('Preparing for sequential prefill');

    fullReset();

    _logicalPos = 0;
    _keepPrefix = 0;
  }

  void prefillFromPosition(int startPos, List<int> tokens) {
    if (_ctx == null) {
      _logger.warning('No context for prefill');
      return;
    }

    _logicalPos = startPos;

    _logger.info('Prefill prepared: startPos=$startPos, tokens=${tokens.length}');
  }

  @override
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _logger.debug('KVCacheManager disposed');
  }

  @override
  bool get isDisposed => _disposed;
}
