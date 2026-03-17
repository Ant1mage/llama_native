/// Llama 错误类型
enum LlamaErrorType { modelLoad, tokenize, contextInit, sessionState, inference, backendInit, kvCache }

/// Llama 统一异常
class LlamaException implements Exception {
  final LlamaErrorType type;
  final String message;
  final Map<String, dynamic>? details;

  const LlamaException(this.type, this.message, {this.details});

  factory LlamaException.modelLoad(String message, {String? filePath}) =>
      LlamaException(LlamaErrorType.modelLoad, message, details: filePath != null ? {'filePath': filePath} : null);

  factory LlamaException.tokenize(String message, {String? text}) =>
      LlamaException(LlamaErrorType.tokenize, message, details: text != null ? {'text': text} : null);

  factory LlamaException.contextInit(String message) => LlamaException(LlamaErrorType.contextInit, message);

  factory LlamaException.sessionState(String message) => LlamaException(LlamaErrorType.sessionState, message);

  factory LlamaException.inference(String message, {int? tokenIndex}) => LlamaException(
    LlamaErrorType.inference,
    message,
    details: tokenIndex != null ? {'tokenIndex': tokenIndex} : null,
  );

  factory LlamaException.backendInit(String message, {String? platform}) =>
      LlamaException(LlamaErrorType.backendInit, message, details: platform != null ? {'platform': platform} : null);

  factory LlamaException.kvCache(String message, {int? currentSize, int? maxSize}) {
    final details = <String, dynamic>{};
    if (currentSize != null) details['currentSize'] = currentSize;
    if (maxSize != null) details['maxSize'] = maxSize;
    return LlamaException(LlamaErrorType.kvCache, message, details: details.isEmpty ? null : details);
  }

  @override
  String toString() {
    final buffer = StringBuffer('LlamaException[${type.name}]: $message');
    if (details != null && details!.isNotEmpty) {
      buffer.write(' (');
      final entries = details!.entries.toList();
      for (var i = 0; i < entries.length; i++) {
        if (i > 0) buffer.write(', ');
        buffer.write('${entries[i].key}: ${entries[i].value}');
      }
      buffer.write(')');
    }
    return buffer.toString();
  }
}
