import 'kv_cache_manager.dart';
import 'package:llama_native/src/log/logger.dart';

/// KV Cache 快照 (简化版)
class KVCacheSnapshot {
  final Logger _logger = Logger('KVCacheSnapshot');

  final int nPast;
  final int keepPrefix;
  final int? windowSize;
  final List<int> data;

  KVCacheSnapshot({required this.nPast, required this.keepPrefix, this.windowSize, required this.data});

  /// 从上下文创建快照
  factory KVCacheSnapshot.fromContext(KVCacheManager manager) {
    // 简化实现：只保存元数据
    // 实际实现应该从 llama_context 获取 KV cache 数据
    return KVCacheSnapshot(nPast: manager.nPast, keepPrefix: manager.keepPrefix, windowSize: null, data: const []);
  }

  /// 恢复到上下文
  void restoreTo(KVCacheManager manager) {
    if (manager.isDisposed) {
      throw StateError('KVCacheManager is disposed');
    }

    _logger.info('Restoring KV cache snapshot: n_past=$nPast, keep_prefix=$keepPrefix');

    // 重置并恢复状态
    manager.reset();
    manager.setKeepPrefix(keepPrefix);
    manager.addProcessed(nPast);

    _logger.debug('KV cache restored successfully');
  }
}
