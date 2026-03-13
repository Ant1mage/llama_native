/// Llama 模型加载异常
class LlamaModelLoadException implements Exception {
  final String message;
  LlamaModelLoadException(this.message);

  @override
  String toString() => 'LlamaModelLoadException: $message';
}

/// Llama Tokenize 异常
class LlamaTokenizeException implements Exception {
  final String message;
  LlamaTokenizeException(this.message);

  @override
  String toString() => 'LlamaTokenizeException: $message';
}

/// Llama 上下文初始化异常
class LlamaContextInitException implements Exception {
  final String message;
  LlamaContextInitException(this.message);

  @override
  String toString() => 'LlamaContextInitException: $message';
}

/// Llama 会话状态异常
class LlamaSessionStateException implements Exception {
  final String message;
  LlamaSessionStateException(this.message);

  @override
  String toString() => 'LlamaSessionStateException: $message';
}
