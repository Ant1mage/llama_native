import 'package:llama_native/src/log/logger.dart';
import 'package:llama_native/src/utils/disposable.dart';

/// 数据流水线 (简化版)
class LlamaBatch with Disposable {
  final Logger _logger = Logger('LlamaBatch');
  bool _disposed = false;

  /// Token 列表 (高级封装，不直接暴露 C 指针)
  final List<int> tokens;

  /// 是否请求所有 logits
  final bool logitsAll;

  /// 创建 batch
  LlamaBatch(this.tokens, {this.logitsAll = false});

  /// 创建单个序列的 batch
  LlamaBatch.single(List<int> tokens, {bool logitsAll = false}) : tokens = tokens, logitsAll = logitsAll;

  @override
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _logger.debug('Batch disposed');
  }

  @override
  bool get isDisposed => _disposed;
}
