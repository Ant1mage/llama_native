/// JSON Schema 到 GBNF Grammar 转换器
///
/// 将 JSON Schema 转换为 llama.cpp 的 GBNF grammar 格式
/// 参考：https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
class JsonSchemaToGbnf {
  final Map<String, String> _rules = {};

  /// 将 JSON Schema 转换为 GBNF grammar 字符串
  String convert(Map<String, dynamic> schema, {String rootName = 'root'}) {
    _rules.clear();

    _convertSchema(schema, rootName);

    final buffer = StringBuffer();
    for (final entry in _rules.entries) {
      buffer.writeln('${entry.key} ::= ${entry.value}');
    }

    return buffer.toString();
  }

  /// 转换 schema 节点
  String _convertSchema(Map<String, dynamic> schema, String name) {
    if (_rules.containsKey(name)) {
      return name;
    }

    final type = _getType(schema);

    switch (type) {
      case 'object':
        return _convertObject(schema, name);
      case 'array':
        return _convertArray(schema, name);
      case 'string':
        return _convertString(schema, name);
      case 'number':
      case 'integer':
        return _convertNumber(schema, name, type);
      case 'boolean':
        return _convertBoolean(name);
      case 'null':
        return _convertNull(name);
      case 'union':
        return _convertUnion(schema, name);
      case 'const':
        return _convertConst(schema['const'], name);
      default:
        return _convertAny(name);
    }
  }

  /// 获取类型
  String _getType(Map<String, dynamic> schema) {
    if (schema.containsKey('type')) {
      final type = schema['type'];
      if (type is List) {
        return type.first as String;
      }
      return type as String;
    }

    if (schema.containsKey('enum')) {
      return 'enum';
    }

    if (schema.containsKey('oneOf') || schema.containsKey('anyOf')) {
      return 'union';
    }

    if (schema.containsKey('const')) {
      return 'const';
    }

    return 'object';
  }

  /// 转换 object 类型
  String _convertObject(Map<String, dynamic> schema, String name) {
    final properties = schema['properties'] as Map<String, dynamic>? ?? {};
    final required = (schema['required'] as List<dynamic>?)?.cast<String>() ?? [];

    if (properties.isEmpty) {
      _rules[name] = '"{" space "}" space';
      return name;
    }

    final parts = <String>[];
    parts.add('"{" space');

    final sortedKeys = properties.keys.toList();
    for (var i = 0; i < sortedKeys.length; i++) {
      final key = sortedKeys[i];
      final propSchema = properties[key] as Map<String, dynamic>;
      final isRequired = required.contains(key);
      final propRuleName = '${name}_${_sanitizeName(key)}';

      final propGbnf = _convertSchema(propSchema, propRuleName);

      if (i > 0) {
        parts.add('"," space');
      }

      final escapedKey = _escapeString(key);
      if (isRequired) {
        parts.add('"\\"$escapedKey\\"" space ":" space $propGbnf');
      } else {
        final optionalRuleName = '${propRuleName}_optional';
        _rules[optionalRuleName] = '"\\"$escapedKey\\"" space ":" space $propGbnf | ""';
        parts.add(optionalRuleName);
      }
    }

    parts.add('"}" space');
    _rules[name] = parts.join(' ');
    return name;
  }

  /// 转换 array 类型
  String _convertArray(Map<String, dynamic> schema, String name) {
    final items = schema['items'] as Map<String, dynamic>?;
    final minItems = schema['minItems'] as int? ?? 0;

    if (items == null) {
      _rules[name] = '"[" space (value ("," space value)*)? "]" space';
      return name;
    }

    final itemRuleName = '${name}_item';
    final itemGbnf = _convertSchema(items, itemRuleName);

    if (minItems > 0) {
      final parts = <String>[];
      parts.add('"[" space');
      parts.add(itemGbnf);
      parts.add('("," space $itemGbnf)*');
      parts.add('"]" space');
      _rules[name] = parts.join(' ');
    } else {
      final arrayItemRule = '${name}_items';
      _rules[arrayItemRule] = '$itemGbnf ("," space $itemGbnf)*';
      _rules[name] = '"[" space ($arrayItemRule)? "]" space';
    }

    return name;
  }

  /// 转换 string 类型
  String _convertString(Map<String, dynamic> schema, String name) {
    if (schema.containsKey('enum')) {
      return _convertEnum(schema, name);
    }

    if (schema.containsKey('const')) {
      final constValue = schema['const'] as String;
      _rules[name] = '"\\"${_escapeString(constValue)}\\"" space';
      return name;
    }

    final pattern = schema['pattern'] as String?;
    if (pattern != null) {
      return _convertPattern(pattern, name);
    }

    final minLength = schema['minLength'] as int?;
    final maxLength = schema['maxLength'] as int?;

    if (minLength != null || maxLength != null) {
      return _convertStringWithLength(name, minLength, maxLength);
    }

    _rules[name] = 'string';
    return name;
  }

  /// 转换带长度限制的字符串
  String _convertStringWithLength(String name, int? minLength, int? maxLength) {
    if (minLength != null && maxLength != null && minLength == maxLength) {
      _rules[name] = '"\\"" [^"\\\\]{$minLength} "\\"" space';
    } else if (minLength != null && maxLength != null) {
      _rules[name] = '"\\"" [^"\\\\]{$minLength,$maxLength} "\\"" space';
    } else if (minLength != null) {
      _rules[name] = '"\\"" [^"\\\\]{$minLength,} "\\"" space';
    } else {
      _rules[name] = '"\\"" [^"\\\\]{0,$maxLength} "\\"" space';
    }
    return name;
  }

  /// 转换 pattern
  String _convertPattern(String pattern, String name) {
    _rules[name] = '"\\"" pattern_$name "\\"" space';
    _rules['pattern_$name'] = _regexToGbnf(pattern);
    return name;
  }

  /// 简化的正则转 GBNF（只支持基本模式）
  String _regexToGbnf(String pattern) {
    var result = pattern;
    result = result.replaceAll('\\d', '[0-9]');
    result = result.replaceAll('\\w', '[a-zA-Z0-9_]');
    result = result.replaceAll('\\s', '[ \\t\\n\\r]');
    return result;
  }

  /// 转换 number/integer 类型
  String _convertNumber(Map<String, dynamic> schema, String name, String type) {
    if (schema.containsKey('enum')) {
      return _convertEnum(schema, name);
    }

    if (schema.containsKey('const')) {
      final constValue = schema['const'];
      _rules[name] = '"$constValue" space';
      return name;
    }

    if (type == 'integer') {
      _rules[name] = 'integer';
    } else {
      _rules[name] = 'number';
    }
    return name;
  }

  /// 转换 boolean 类型
  String _convertBoolean(String name) {
    _rules[name] = 'boolean';
    return name;
  }

  /// 转换 null 类型
  String _convertNull(String name) {
    _rules[name] = 'null';
    return name;
  }

  /// 转换 enum 类型
  String _convertEnum(Map<String, dynamic> schema, String name) {
    final values = schema['enum'] as List<dynamic>;
    final escapedValues = values
        .map((v) {
          if (v is String) {
            return '"\\"${_escapeString(v)}\\""';
          } else if (v == null) {
            return 'null';
          } else if (v is bool) {
            return v ? 'true' : 'false';
          } else {
            return '"$v"';
          }
        })
        .join(' | ');

    _rules[name] = '$escapedValues space';
    return name;
  }

  /// 转换 union 类型 (oneOf/anyOf)
  String _convertUnion(Map<String, dynamic> schema, String name) {
    final variants = (schema['oneOf'] ?? schema['anyOf']) as List<dynamic>;
    final variantNames = <String>[];

    for (var i = 0; i < variants.length; i++) {
      final variant = variants[i] as Map<String, dynamic>;
      final variantName = '${name}_variant$i';
      final gbnf = _convertSchema(variant, variantName);
      variantNames.add(gbnf);
    }

    _rules[name] = variantNames.join(' | ');
    return name;
  }

  /// 转换 const 类型
  String _convertConst(dynamic constValue, String name) {
    if (constValue is String) {
      _rules[name] = '"\\"${_escapeString(constValue)}\\"" space';
    } else if (constValue == null) {
      _rules[name] = 'null';
    } else if (constValue is bool) {
      _rules[name] = constValue ? 'true space' : 'false space';
    } else {
      _rules[name] = '"$constValue" space';
    }
    return name;
  }

  /// 转换任意类型
  String _convertAny(String name) {
    _rules[name] = 'value';
    return name;
  }

  /// 清理名称用于规则名
  String _sanitizeName(String name) {
    return name.replaceAll(RegExp(r'[^a-zA-Z0-9_]'), '_');
  }

  /// 转义字符串
  String _escapeString(String s) {
    return s
        .replaceAll('\\', '\\\\')
        .replaceAll('"', '\\"')
        .replaceAll('\n', '\\n')
        .replaceAll('\r', '\\r')
        .replaceAll('\t', '\\t');
  }
}

/// 生成完整的 GBNF grammar（包含基础类型定义）
String generateGbnfGrammar(Map<String, dynamic> schema, {String rootName = 'root'}) {
  final converter = JsonSchemaToGbnf();
  final mainGrammar = converter.convert(schema, rootName: rootName);

  final buffer = StringBuffer();

  buffer.writeln('''
root ::= $rootName

value ::= object | array | string | number | boolean | null

object ::= "{" space (pair ("," space pair)*)? "}" space
pair ::= string ":" space value

array ::= "[" space (value ("," space value)*)? "]" space

string ::= "\\"" char* "\\"" space
char ::= [^"\\\\] | "\\\\" escape
escape ::= ["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]

number ::= integer ("." [0-9]+)? ([eE] [+-]? [0-9]+)? space
integer ::= [0-9]+ | "-" [0-9]+

boolean ::= "true" space | "false" space
null ::= "null" space

space ::= " "*
''');

  buffer.write(mainGrammar);

  return buffer.toString();
}
