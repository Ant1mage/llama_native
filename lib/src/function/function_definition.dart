import 'dart:async';

/// 函数参数定义
class FunctionParameter {
  final String name;
  final String type;
  final String? description;
  final List<String>? enumValues;
  final bool required;

  const FunctionParameter({
    required this.name,
    required this.type,
    this.description,
    this.enumValues,
    this.required = true,
  });

  Map<String, dynamic> toJsonSchema() {
    final schema = <String, dynamic>{'type': type};
    if (description != null) schema['description'] = description;
    if (enumValues != null) schema['enum'] = enumValues;
    return schema;
  }
}

/// 函数定义
class FunctionDefinition {
  final String name;
  final String description;
  final List<FunctionParameter> parameters;
  final Future<String> Function(Map<String, dynamic> params) handler;

  const FunctionDefinition({
    required this.name,
    required this.description,
    required this.parameters,
    required this.handler,
  });

  Map<String, dynamic> toJsonSchema() {
    final props = <String, dynamic>{};
    final required = <String>[];

    for (final param in parameters) {
      props[param.name] = param.toJsonSchema();
      if (param.required) required.add(param.name);
    }

    return {'type': 'object', 'properties': props, 'required': required};
  }

  Map<String, dynamic> toOpenAiSchema() {
    return {
      'type': 'function',
      'function': {'name': name, 'description': description, 'parameters': toJsonSchema()},
    };
  }
}

/// 函数调用结果
class FunctionCallResult {
  final String name;
  final Map<String, dynamic> arguments;
  final String? result;
  final String? error;

  const FunctionCallResult({required this.name, required this.arguments, this.result, this.error});

  bool get hasError => error != null;
  bool get hasResult => result != null;

  @override
  String toString() {
    if (hasError) return 'FunctionCallResult($name: ERROR $error)';
    return 'FunctionCallResult($name: $result)';
  }
}

/// 函数调用解析器
///
/// 从模型输出中解析函数调用
class FunctionCallParser {
  static final RegExp _functionCallPattern = RegExp(
    r'<function_call>\s*(\{[\s\S]*?\})\s*</function_call>',
    multiLine: true,
  );

  static final RegExp _jsonBlockPattern = RegExp(r'```json\s*(\{[\s\S]*?\})\s*```', multiLine: true);

  static final RegExp _toolUsePattern = RegExp(r'<tool_use>\s*(\{[\s\S]*?\})\s*</tool_use>', multiLine: true);

  /// 尝试解析函数调用
  static List<({String name, Map<String, dynamic> args})>? parse(String text) {
    final results = <({String name, Map<String, dynamic> args})>[];

    for (final pattern in [_functionCallPattern, _jsonBlockPattern, _toolUsePattern]) {
      final matches = pattern.allMatches(text);
      for (final match in matches) {
        final jsonStr = match.group(1);
        if (jsonStr == null) continue;

        try {
          final parsed = _parseJson(jsonStr);
          if (parsed != null) {
            final name = parsed['name'] as String? ?? parsed['function'] as String?;
            final args =
                parsed['arguments'] as Map<String, dynamic>? ?? parsed['parameters'] as Map<String, dynamic>? ?? {};

            if (name != null) {
              results.add((name: name, args: Map<String, dynamic>.from(args)));
            }
          }
        } catch (_) {
          continue;
        }
      }
    }

    if (results.isEmpty) {
      final directCall = _tryParseDirectCall(text);
      if (directCall != null) {
        results.add(directCall);
      }
    }

    return results.isEmpty ? null : results;
  }

  static Map<String, dynamic>? _parseJson(String jsonStr) {
    try {
      final decoded = _parseJsonWithRetry(jsonStr);
      return decoded as Map<String, dynamic>;
    } catch (_) {
      return null;
    }
  }

  static dynamic _parseJsonWithRetry(String jsonStr) {
    var cleaned = jsonStr.trim();

    if (!cleaned.startsWith('{') && !cleaned.startsWith('[')) {
      final startIndex = cleaned.indexOf('{');
      final startIndexArray = cleaned.indexOf('[');
      if (startIndex >= 0 && (startIndexArray < 0 || startIndex < startIndexArray)) {
        cleaned = cleaned.substring(startIndex);
      } else if (startIndexArray >= 0) {
        cleaned = cleaned.substring(startIndexArray);
      }
    }

    if (cleaned.startsWith('{') && !cleaned.endsWith('}')) {
      final lastBrace = cleaned.lastIndexOf('}');
      if (lastBrace > 0) {
        cleaned = cleaned.substring(0, lastBrace + 1);
      }
    }

    return _simpleJsonDecode(cleaned);
  }

  static dynamic _simpleJsonDecode(String jsonStr) {
    final buffer = StringBuffer();
    bool inString = false;
    bool escaped = false;

    for (var i = 0; i < jsonStr.length; i++) {
      final char = jsonStr[i];

      if (escaped) {
        buffer.write(char);
        escaped = false;
        continue;
      }

      if (char == '\\' && inString) {
        buffer.write(char);
        escaped = true;
        continue;
      }

      if (char == '"') {
        inString = !inString;
      }

      if (!inString && (char == '\n' || char == '\r' || char == '\t')) {
        continue;
      }

      buffer.write(char);
    }

    return _parseJsonValue(buffer.toString(), 0).$1;
  }

  static (dynamic, int) _parseJsonValue(String s, int start) {
    var i = start;
    while (i < s.length && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) {
      i++;
    }

    if (i >= s.length) return (null, i);

    if (s[i] == '{') {
      return _parseJsonObject(s, i);
    } else if (s[i] == '[') {
      return _parseJsonArray(s, i);
    } else if (s[i] == '"') {
      return _parseJsonString(s, i);
    } else if (s[i] == 't' || s[i] == 'f') {
      return _parseJsonBool(s, i);
    } else if (s[i] == 'n') {
      return _parseJsonNull(s, i);
    } else if (s[i] == '-' || (s.codeUnitAt(i) >= 48 && s.codeUnitAt(i) <= 57)) {
      return _parseJsonNumber(s, i);
    }

    return (null, i);
  }

  static (Map<String, dynamic>?, int) _parseJsonObject(String s, int start) {
    final result = <String, dynamic>{};
    var i = start + 1;

    while (i < s.length && s[i] != '}') {
      while (i < s.length && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t' || s[i] == ',')) {
        i++;
      }
      if (i >= s.length || s[i] == '}') break;

      final (key, keyEnd) = _parseJsonString(s, i);
      if (key == null) break;
      i = keyEnd;

      while (i < s.length && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) {
        i++;
      }
      if (i >= s.length || s[i] != ':') break;
      i++;

      final (value, valueEnd) = _parseJsonValue(s, i);
      result[key] = value;
      i = valueEnd;
    }

    if (i < s.length && s[i] == '}') i++;
    return (result, i);
  }

  static (List<dynamic>?, int) _parseJsonArray(String s, int start) {
    final result = <dynamic>[];
    var i = start + 1;

    while (i < s.length && s[i] != ']') {
      while (i < s.length && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t' || s[i] == ',')) {
        i++;
      }
      if (i >= s.length || s[i] == ']') break;

      final (value, valueEnd) = _parseJsonValue(s, i);
      result.add(value);
      i = valueEnd;
    }

    if (i < s.length && s[i] == ']') i++;
    return (result, i);
  }

  static (String?, int) _parseJsonString(String s, int start) {
    if (s[start] != '"') return (null, start);
    var i = start + 1;
    final buffer = StringBuffer();

    while (i < s.length && s[i] != '"') {
      if (s[i] == '\\' && i + 1 < s.length) {
        i++;
        switch (s[i]) {
          case 'n':
            buffer.write('\n');
            break;
          case 'r':
            buffer.write('\r');
            break;
          case 't':
            buffer.write('\t');
            break;
          case '"':
            buffer.write('"');
            break;
          case '\\':
            buffer.write('\\');
            break;
          default:
            buffer.write(s[i]);
        }
      } else {
        buffer.write(s[i]);
      }
      i++;
    }

    if (i < s.length && s[i] == '"') i++;
    return (buffer.toString(), i);
  }

  static (bool?, int) _parseJsonBool(String s, int start) {
    if (s.startsWith('true', start)) {
      return (true, start + 4);
    } else if (s.startsWith('false', start)) {
      return (false, start + 5);
    }
    return (null, start);
  }

  static (Null, int) _parseJsonNull(String s, int start) {
    if (s.startsWith('null', start)) {
      return (null, start + 4);
    }
    return (null, start);
  }

  static (num?, int) _parseJsonNumber(String s, int start) {
    var i = start;
    if (s[i] == '-') i++;

    while (i < s.length && s.codeUnitAt(i) >= 48 && s.codeUnitAt(i) <= 57) {
      i++;
    }

    if (i < s.length && s[i] == '.') {
      i++;
      while (i < s.length && s.codeUnitAt(i) >= 48 && s.codeUnitAt(i) <= 57) {
        i++;
      }
    }

    if (i < s.length && (s[i] == 'e' || s[i] == 'E')) {
      i++;
      if (i < s.length && (s[i] == '+' || s[i] == '-')) i++;
      while (i < s.length && s.codeUnitAt(i) >= 48 && s.codeUnitAt(i) <= 57) {
        i++;
      }
    }

    final numStr = s.substring(start, i);
    final parsed = numStr.contains('.') || numStr.contains('e') || numStr.contains('E')
        ? double.parse(numStr)
        : int.parse(numStr);
    return (parsed, i);
  }

  static ({String name, Map<String, dynamic> args})? _tryParseDirectCall(String text) {
    final namePattern = RegExp(r'"name"\s*:\s*"([^"]+)"');
    final argsPattern = RegExp(r'"arguments"\s*:\s*(\{[\s\S]*?\})');

    final nameMatch = namePattern.firstMatch(text);
    final argsMatch = argsPattern.firstMatch(text);

    if (nameMatch != null) {
      final name = nameMatch.group(1)!;
      Map<String, dynamic> args = {};

      if (argsMatch != null) {
        try {
          final argsJson = argsMatch.group(1)!;
          final parsed = _parseJson(argsJson);
          if (parsed != null) {
            args = parsed;
          }
        } catch (_) {}
      }

      return (name: name, args: args);
    }

    return null;
  }
}
