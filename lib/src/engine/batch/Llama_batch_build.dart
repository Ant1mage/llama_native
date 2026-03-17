import 'package:llama_native/src/engine/batch/llama_batch.dart';

/// 批处理构建器
class LlamaBatchBuilder {
  final List<List<int>> _sequences = [];
  bool _logitsAll = false;

  /// 添加序列
  void addSequence(List<int> tokens) {
    _sequences.add(tokens);
  }

  /// 设置是否请求所有 logits
  void setLogitsAll(bool value) {
    _logitsAll = value;
  }

  /// 构建 batch
  LlamaBatch build() {
    if (_sequences.isEmpty) {
      throw StateError('Cannot build empty batch');
    }

    // 合并所有 sequences
    final allTokens = <int>[];
    for (final seq in _sequences) {
      allTokens.addAll(seq);
    }

    return LlamaBatch(allTokens, logitsAll: _logitsAll);
  }

  /// 清空构建器
  void clear() {
    _sequences.clear();
    _logitsAll = false;
  }
}
