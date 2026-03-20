import 'dart:async';
import 'dart:convert';
import 'dart:ffi';

import 'package:llama_native/src/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/engine/function/function_definition.dart';
import 'package:llama_native/src/engine/grammar/grammar.dart';
import 'package:llama_native/src/log/logger.dart';

class FunctionManager {
  final Map<String, FunctionDefinition> _functions = {};
  final Logger _logger = Logger('FunctionManager');

  void register(FunctionDefinition function) {
    _functions[function.name] = function;
    _logger.debug('注册函数: ${function.name}');
  }

  void registerAll(List<FunctionDefinition> functions) {
    for (final func in functions) {
      register(func);
    }
  }

  void unregister(String name) {
    _functions.remove(name);
    _logger.debug('注销函数: $name');
  }

  FunctionDefinition? getFunction(String name) => _functions[name];

  List<String> get functionNames => _functions.keys.toList();

  List<FunctionDefinition> get functions => _functions.values.toList();

  void clear() {
    _functions.clear();
    _logger.debug('清空所有函数');
  }

  List<Map<String, dynamic>> toOpenAiSchema() {
    return _functions.values.map((f) => f.toOpenAiSchema()).toList();
  }

  Map<String, dynamic> generateFunctionCallSchema() {
    final functionSchemas = <Map<String, dynamic>>[];

    for (final func in _functions.values) {
      functionSchemas.add({
        'type': 'object',
        'properties': {
          'name': {'type': 'string', 'const': func.name},
          'arguments': func.toJsonSchema(),
        },
        'required': ['name', 'arguments'],
      });
    }

    return {
      'type': 'object',
      'properties': {
        'function_call': {
          'type': 'object',
          'properties': {
            'name': {'type': 'string', 'enum': _functions.keys.toList()},
            'arguments': {'type': 'object'},
          },
          'required': ['name', 'arguments'],
        },
      },
      'required': ['function_call'],
    };
  }

  Map<String, dynamic> generateFunctionListSchema() {
    return {'type': 'array', 'items': generateFunctionCallSchema()['properties']['function_call']};
  }

  Grammar createGrammar(Pointer<bindings.llama_vocab> vocab) {
    final schema = generateFunctionCallSchema();
    return Grammar.fromJsonSchema(schema, vocab, rootRule: 'root');
  }

  Future<List<FunctionCallResult>> parseAndExecute(String modelOutput) async {
    final results = <FunctionCallResult>[];

    final calls = FunctionCallParser.parse(modelOutput);
    if (calls == null || calls.isEmpty) {
      _logger.debug('输出中未找到函数调用');
      return results;
    }

    for (final call in calls) {
      final func = _functions[call.name];
      if (func == null) {
        _logger.warning('未知函数: ${call.name}');
        results.add(FunctionCallResult(name: call.name, arguments: call.args, error: '未知函数: ${call.name}'));
        continue;
      }

      try {
        _logger.info('执行函数: ${call.name}，参数: ${call.args}');
        final result = await func.handler(call.args);
        results.add(FunctionCallResult(name: call.name, arguments: call.args, result: result));
        _logger.debug('函数${call.name}返回: $result');
      } catch (e) {
        _logger.error('函数${call.name}执行失败: $e');
        results.add(FunctionCallResult(name: call.name, arguments: call.args, error: e.toString()));
      }
    }

    return results;
  }

  String formatResultsAsMessage(List<FunctionCallResult> results) {
    final buffer = StringBuffer();

    for (final result in results) {
      buffer.writeln('函数: ${result.name}');
      if (result.hasError) {
        buffer.writeln('错误: ${result.error}');
      } else if (result.hasResult) {
        buffer.writeln('结果: ${result.result}');
      }
      buffer.writeln('---');
    }

    return buffer.toString();
  }
}

class FunctionCallingHelper {
  static String generateSystemPrompt(List<FunctionDefinition> functions) {
    final buffer = StringBuffer();

    buffer.writeln('你可以使用以下函数:');
    buffer.writeln();

    for (final func in functions) {
      buffer.writeln('## ${func.name}');
      buffer.writeln(func.description);
      buffer.writeln('参数:');
      for (final param in func.parameters) {
        final required = param.required ? ' (必填)' : '';
        buffer.writeln('  - ${param.name}$required: ${param.type}');
        if (param.description != null) {
          buffer.writeln('    ${param.description}');
        }
        if (param.enumValues != null) {
          buffer.writeln('    允许的值: ${param.enumValues!.join(', ')}');
        }
      }
      buffer.writeln();
    }

    buffer.writeln('调用函数时，请使用以下JSON格式响应:');
    buffer.writeln('```json');
    buffer.writeln('{');
    buffer.writeln('  "name": "函数名",');
    buffer.writeln('  "arguments": {');
    buffer.writeln('    "参数1": "值1",');
    buffer.writeln('    "参数2": "值2"');
    buffer.writeln('  }');
    buffer.writeln('}');
    buffer.writeln('```');
    buffer.writeln();
    buffer.writeln('仅在必要时调用函数。如果不需要调用函数，请正常响应。');

    return buffer.toString();
  }

  static String generateLlama3Prompt(List<FunctionDefinition> functions) {
    final toolsJson = functions.map((f) => f.toOpenAiSchema()).toList();
    return '''
可用工具:
${jsonEncode(toolsJson)}

需要调用函数时，请使用以下格式输出:
<|start_header_id|>assistant<|end_header_id|>

{"name": "函数名", "arguments": {"参数1": "值1"}}
''';
  }

  static String generateQwenPrompt(List<FunctionDefinition> functions) {
    final toolsJson = functions.map((f) => f.toOpenAiSchema()).toList();
    return '''
# 工具

你可以调用一个或多个函数来辅助回答用户问题。
函数描述格式如下:

${_jsonEncode(toolsJson)}

每次函数调用，请在以下XML标签内输出函数名和参数:
<tool_call name="函数名">
{"参数1": "值1"}
</tool_call >
''';
  }
}

String _jsonEncode(dynamic obj) {
  return JsonEncoder.withIndent('  ').convert(obj);
}
