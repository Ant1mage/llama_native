import 'dart:ffi';
import 'package:llama_native/src/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/log/logger.dart';

import 'package:llama_native/src/engine/exceptions/llama_exceptions.dart';
import 'package:llama_native/src/engine/context/token_generation.dart';

typedef KVCacheRebuildCallback = List<int> Function(int neededTokens);
typedef KVCacheThresholdCallback = void Function(double usagePercent);
typedef KVCacheResetCallback = void Function();
typedef KVCacheSnapshotCallback = Future<String?> Function(double usagePercent);

class KVCacheManager {
  final Logger _logger = Logger('LlamaKVCacheManager');
  final int _nCtx;
  int _logicalPos = 0;
  int _keepPrefix = 0;
  int _nRecent = 256;
  int? _windowSize;
  Pointer<bindings.llama_context>? _ctx;
  KVCacheRebuildCallback? _rebuildCallback;
  KVCacheThresholdCallback? _onNearThreshold;
  KVCacheThresholdCallback? _onEmergency;
  KVCacheResetCallback? _onFullReset;
  KVCacheSnapshotCallback? _onSnapshotNeeded;

  bool _isPaused = false;
  bool _snapshotInjected = false;

  static const double _moeBufferPercent = 0.20;
  static const double _truncationThreshold = 0.75;
  static const double _emergencyThreshold = 0.80;

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

  void setOnNearThreshold(KVCacheThresholdCallback? callback) {
    _onNearThreshold = callback;
  }

  void setOnEmergency(KVCacheThresholdCallback? callback) {
    _onEmergency = callback;
  }

  void setOnFullReset(KVCacheResetCallback? callback) {
    _onFullReset = callback;
  }

  void setOnSnapshotNeeded(KVCacheSnapshotCallback? callback) {
    _onSnapshotNeeded = callback;
  }

  bool get isPaused => _isPaused;
  bool get snapshotInjected => _snapshotInjected;

  void pause() {
    _isPaused = true;
    _logger.info('KV Cache 管理已暂停');
  }

  void resume() {
    _isPaused = false;
    _logger.info('KV Cache 管理已恢复');
  }

  void markSnapshotInjected() {
    _snapshotInjected = true;
    _logger.info('快照已注入标记');
  }

  int get nPast => _logicalPos;
  int get nRemain => _nCtx - _usedCells;
  int get keepPrefix => _keepPrefix;
  int get logicalPos => _logicalPos;

  void setKeepPrefix(int prefixTokens) {
    _keepPrefix = prefixTokens;
    _logger.debug('设置保留前缀: $prefixTokens');
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
    if (_isPaused) {
      _logger.debug('KV Cache 管理已暂停，跳过检查');
      return;
    }

    final usage = usagePercent;

    if (isFull) {
      _logger.warning('KV缓存已满: $_usedCells/$_nCtx (${usage.toStringAsFixed(1)}%)');
      _handleFullCache();
    } else if (isOverSafeThreshold) {
      _logger.warning('KV缓存超过安全阈值: $_usedCells/$_nCtx (${usage.toStringAsFixed(1)}%)');
      _onEmergency?.call(usage);
      _handleOverThreshold();
    } else if (needsTruncation) {
      _logger.info('KV缓存接近阈值: $_usedCells/$_nCtx (${usage.toStringAsFixed(1)}%)');
      _onNearThreshold?.call(usage);
      if (!_snapshotInjected) {
        _logger.info('等待快照注入后再截断...');
        pause();
      } else {
        _autoTruncate();
      }
    } else if (_windowSize != null && _usedCells > _windowSize!) {
      _logger.debug('触发滑动窗口');
      _applySlidingWindow();
    }
  }

  void _handleFullCache() {
    _logger.error('KV缓存耗尽，无法继续推理');
    throw LlamaException.kvCache('KV cache full, inference terminated');
  }

  void _handleOverThreshold() {
    if (_rebuildCallback != null) {
      _logger.info('请求回调重建缓存');
      try {
        final tokensToRestore = _rebuildCallback!(_nCtx ~/ 4);
        if (tokensToRestore.isNotEmpty) {
          requestRebuild(tokensToRestore);
          return;
        }
      } catch (e) {
        _logger.error('重建回调失败: $e');
      }
    }

    _logger.warning('执行紧急截断');
    _emergencyTruncate();
  }

  void requestRebuild(List<int> tokens) {
    if (_ctx == null) {
      _logger.warning('无法重建: 无上下文');
      return;
    }

    try {
      final mem = bindings.llama_get_memory(_ctx!);
      bindings.llama_memory_clear(mem, true);
      bindings.llama_synchronize(_ctx!);
      _logicalPos = 0;
      _logger.info('KV缓存已清空准备重建，可容纳${tokens.length}个Token');
    } catch (e) {
      _logger.error('清空缓存准备重建时出错: $e');
    }
  }

  void _emergencyTruncate() {
    if (_ctx == null) {
      _logger.warning('无上下文无法截断');
      return;
    }

    try {
      final mem = bindings.llama_get_memory(_ctx!);
      final usedCells = _usedCells;

      final removeStart = _keepPrefix;
      final removeEnd = usedCells - _nRecent;

      if (removeEnd <= removeStart) {
        _logger.warning('无法截断: 保留前缀=$_keepPrefix, 最近N个=$_nRecent, 已用=$usedCells');
        _logger.warning('回退到完全清空');
        clear();
        return;
      }

      final removed = bindings.llama_memory_seq_rm(mem, 0, removeStart, removeEnd);

      if (removed) {
        bindings.llama_synchronize(_ctx!);
        final newUsed = _usedCells;
        _logger.info('紧急截断: 移除位置 $removeStart-$removeEnd');
        _logger.info('已用单元: $usedCells -> $newUsed (${(newUsed / _nCtx * 100).toStringAsFixed(1)}%)');

        if (newUsed > _safeCapacity) {
          _logger.error('截断后仍超过安全阈值，请求完全重置');
          throw LlamaException.kvCache('KV cache cannot be recovered, full reset required');
        }
      } else {
        _logger.warning('紧急截断失败，清空缓存');
        clear();
      }
    } catch (e) {
      if (e is LlamaException) rethrow;
      _logger.error('紧急截断时出错: $e');
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
        _logger.warning('无效截断范围: $removeStart-$removeEnd (已用=$usedCells)');
        return;
      }

      final removed = bindings.llama_memory_seq_rm(mem, 0, removeStart, removeEnd);

      if (removed) {
        bindings.llama_synchronize(_ctx!);
        final newUsed = _usedCells;
        _logger.info('自动截断: 移除$removeCount个单元，使用量: ${usedCells} -> $newUsed');
      } else {
        _logger.warning('自动截断失败');
      }
    } catch (e) {
      _logger.error('自动截断时出错: $e');
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
        _logger.info('滑动窗口: 移除$removeCount个单元');
      }
    } catch (e) {
      _logger.error('滑动窗口时出错: $e');
    }
  }

  void clear() {
    if (_ctx != null) {
      try {
        final mem = bindings.llama_get_memory(_ctx!);
        bindings.llama_memory_clear(mem, true);
        bindings.llama_synchronize(_ctx!);
        _logger.info('KV缓存已完全清空');
      } catch (e) {
        _logger.error('清空缓存时出错: $e');
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
        _logger.debug('从上下文同步逻辑位置: $_logicalPos');
      }
    } catch (e) {
      _logger.warning('从上下文同步失败: $e');
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
      _logger.error('MoE缓冲区溢出: 预计=$projectedUsage, 安全容量=$_safeCapacity');
      throw LlamaException.kvCache('MoE buffer overflow, need reset before continuing');
    }
  }

  void fullReset() {
    if (_ctx == null) {
      _logger.warning('无上下文无法完全重置');
      _logicalPos = 0;
      return;
    }

    _logger.info('执行KV缓存完全重置');

    try {
      final mem = bindings.llama_get_memory(_ctx!);

      bindings.llama_memory_seq_rm(mem, -1, 0, -1);

      bindings.llama_synchronize(_ctx!);

      _logicalPos = 0;

      _onFullReset?.call();

      _logger.info('KV缓存已完全重置: 已用=$_usedCells, 逻辑位置=$_logicalPos');
    } catch (e) {
      _logger.error('完全重置时出错: $e');
      _logicalPos = 0;
    }
  }

  void injectContextAfterTruncation() {
    _logger.info('注入上下文快照后恢复管理');
    _snapshotInjected = true;
    _isPaused = false;
    _autoTruncate();
    _logger.info('上下文注入完成，管理已恢复');
  }

  void prepareForSnapshot() {
    _logger.info('准备接收快照注入');
    _snapshotInjected = false;
  }

  void prepareSequentialPrefill() {
    if (_ctx == null) {
      _logger.warning('无上下文无法准备顺序预填充');
      return;
    }

    _logger.info('准备顺序预填充');

    fullReset();

    _logicalPos = 0;
    _keepPrefix = 0;
  }

  void prefillFromPosition(int startPos, List<int> tokens) {
    if (_ctx == null) {
      _logger.warning('无上下文无法预填充');
      return;
    }

    _logicalPos = startPos;

    _logger.info('预填充准备完成: 起始位置=$startPos, Token数=${tokens.length}');
  }

  void dispose() {
    if (_ctx == null) return;
    _ctx = null;
    _logger.debug('KVCacheManager已释放');
  }

  bool get isDisposed => _ctx == null;
}
