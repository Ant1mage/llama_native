import 'kv_cache_manager.dart';

/// KV Cache 快照 (简化版)
class KVCacheSnapshot {
  final int nPast;
  final int keepPrefix;
  final int? windowSize;
  final List<int> data;

  KVCacheSnapshot({required this.nPast, required this.keepPrefix, this.windowSize, required this.data});

  /// 从上下文创建快照
  factory KVCacheSnapshot.fromContext(KVCacheManager manager) {
    return KVCacheSnapshot(nPast: manager.nPast, keepPrefix: manager.keepPrefix, windowSize: null, data: const []);
  }

  /// 恢复到上下文
  void restoreTo(KVCacheManager manager) {
    // 简化实现
  }
}
