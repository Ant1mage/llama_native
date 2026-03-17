/// Llama 模型加载异常
class LlamaModelLoadException implements Exception {
  final String message;
  final String? filePath;
  
  LlamaModelLoadException(this.message, {this.filePath});

  @override
  String toString() => filePath != null 
      ? 'LlamaModelLoadException: $message (File: $filePath)'
      : 'LlamaModelLoadException: $message';
}

/// Llama Tokenize 异常
class LlamaTokenizeException implements Exception {
  final String message;
  final String? text;
  
  LlamaTokenizeException(this.message, {this.text});

  @override
  String toString() => text != null 
      ? 'LlamaTokenizeException: $message (Text: ${text!.length > 50 ? text!.substring(0, 50) + '...' : text})'
      : 'LlamaTokenizeException: $message';
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

/// Llama 推理异常
class LlamaInferenceException implements Exception {
  final String message;
  final int? tokenIndex;
  
  LlamaInferenceException(this.message, {this.tokenIndex});

  @override
  String toString() => tokenIndex != null 
      ? 'LlamaInferenceException: $message (Token: $tokenIndex)'
      : 'LlamaInferenceException: $message';
}

/// Llama 后端初始化异常
class LlamaBackendInitException implements Exception {
  final String message;
  final String? platform;
  
  LlamaBackendInitException(this.message, {this.platform});

  @override
  String toString() => platform != null 
      ? 'LlamaBackendInitException: $message (Platform: $platform)'
      : 'LlamaBackendInitException: $message';
}

/// Llama KV Cache 异常
class LlamaKVCacheException implements Exception {
  final String message;
  final int? currentSize;
  final int? maxSize;
  
  LlamaKVCacheException(this.message, {this.currentSize, this.maxSize});

  @override
  String toString() {
    if (currentSize != null && maxSize != null) {
      return 'LlamaKVCacheException: $message (Current: $currentSize, Max: $maxSize)';
    }
    return 'LlamaKVCacheException: $message';
  }
}
