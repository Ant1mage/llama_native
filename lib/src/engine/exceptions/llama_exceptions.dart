/// Llama 错误类型
enum LlamaErrorType { model, tokenize, context, session, inference, backend, kvCache }

/// Llama 统一异常
class LlamaException implements Exception {
  final LlamaErrorType type;
  final String message;
  final Map<String, dynamic>? details;

  const LlamaException(this.type, this.message, {this.details});

  factory LlamaException.model(String message, {String? filePath}) =>
      LlamaException(LlamaErrorType.model, message, details: filePath != null ? {'filePath': filePath} : null);

  factory LlamaException.tokenize(String message, {String? text}) =>
      LlamaException(LlamaErrorType.tokenize, message, details: text != null ? {'text': text} : null);

  factory LlamaException.context(String message) => LlamaException(LlamaErrorType.context, message);

  factory LlamaException.session(String message) => LlamaException(LlamaErrorType.session, message);

  factory LlamaException.inference(String message, {int? tokenIndex}) => LlamaException(
    LlamaErrorType.inference,
    message,
    details: tokenIndex != null ? {'tokenIndex': tokenIndex} : null,
  );

  factory LlamaException.backend(String message, {String? platform}) =>
      LlamaException(LlamaErrorType.backend, message, details: platform != null ? {'platform': platform} : null);

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
