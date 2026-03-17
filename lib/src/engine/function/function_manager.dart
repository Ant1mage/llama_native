import 'dart:async';
import 'dart:convert';
import 'dart:ffi';

import 'package:llama_native/src/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/engine/function/function_definition.dart';
import 'package:llama_native/src/engine/grammar/grammar.dart';
import 'package:llama_native/src/log/logger.dart';

/// Function Calling 管理器
///
/// 提供：
/// - 函数注册与管理
/// - 生成函数调用的 JSON Schema
/// - 解析模型输出并执行函数
class FunctionManager {
  final Map<String, FunctionDefinition> _functions = {};
  final Logger _logger = Logger('FunctionManager');

  /// 注册函数
  void register(FunctionDefinition function) {
    _functions[function.name] = function;
    _logger.debug('Registered function: ${function.name}');
  }

  /// 注册多个函数
  void registerAll(List<FunctionDefinition> functions) {
    for (final func in functions) {
      register(func);
    }
  }

  /// 注销函数
  void unregister(String name) {
    _functions.remove(name);
    _logger.debug('Unregistered function: $name');
  }

  /// 获取函数
  FunctionDefinition? getFunction(String name) => _functions[name];

  /// 获取所有已注册的函数名
  List<String> get functionNames => _functions.keys.toList();

  /// 获取所有已注册的函数
  List<FunctionDefinition> get functions => _functions.values.toList();

  /// 清空所有函数
  void clear() {
    _functions.clear();
    _logger.debug('Cleared all functions');
  }

  /// 生成 OpenAI 格式的函数定义
  List<Map<String, dynamic>> toOpenAiSchema() {
    return _functions.values.map((f) => f.toOpenAiSchema()).toList();
  }

  /// 生成函数调用的 JSON Schema（用于 Grammar 约束）
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

  /// 生成函数列表的 JSON Schema（用于多函数调用）
  Map<String, dynamic> generateFunctionListSchema() {
    return {'type': 'array', 'items': generateFunctionCallSchema()['properties']['function_call']};
  }

  /// 创建函数调用的 Grammar
  Grammar createGrammar(Pointer<bindings.llama_vocab> vocab) {
    final schema = generateFunctionCallSchema();
    return Grammar.fromJsonSchema(schema, vocab, rootRule: 'root');
  }

  /// 解析模型输出并执行函数
  Future<List<FunctionCallResult>> parseAndExecute(String modelOutput) async {
    final results = <FunctionCallResult>[];

    final calls = FunctionCallParser.parse(modelOutput);
    if (calls == null || calls.isEmpty) {
      _logger.debug('No function calls found in output');
      return results;
    }

    for (final call in calls) {
      final func = _functions[call.name];
      if (func == null) {
        _logger.warning('Unknown function: ${call.name}');
        results.add(FunctionCallResult(name: call.name, arguments: call.args, error: 'Unknown function: ${call.name}'));
        continue;
      }

      try {
        _logger.info('Executing function: ${call.name} with args: ${call.args}');
        final result = await func.handler(call.args);
        results.add(FunctionCallResult(name: call.name, arguments: call.args, result: result));
        _logger.debug('Function ${call.name} returned: $result');
      } catch (e) {
        _logger.error('Function ${call.name} failed: $e');
        results.add(FunctionCallResult(name: call.name, arguments: call.args, error: e.toString()));
      }
    }

    return results;
  }

  /// 格式化函数调用结果为消息
  String formatResultsAsMessage(List<FunctionCallResult> results) {
    final buffer = StringBuffer();

    for (final result in results) {
      buffer.writeln('Function: ${result.name}');
      if (result.hasError) {
        buffer.writeln('Error: ${result.error}');
      } else if (result.hasResult) {
        buffer.writeln('Result: ${result.result}');
      }
      buffer.writeln('---');
    }

    return buffer.toString();
  }
}

/// 函数调用辅助方法
class FunctionCallingHelper {
  /// 生成函数定义的系统提示
  static String generateSystemPrompt(List<FunctionDefinition> functions) {
    final buffer = StringBuffer();

    buffer.writeln('You have access to the following functions:');
    buffer.writeln();

    for (final func in functions) {
      buffer.writeln('## ${func.name}');
      buffer.writeln(func.description);
      buffer.writeln('Parameters:');
      for (final param in func.parameters) {
        final required = param.required ? ' (required)' : '';
        buffer.writeln('  - ${param.name}$required: ${param.type}');
        if (param.description != null) {
          buffer.writeln('    ${param.description}');
        }
        if (param.enumValues != null) {
          buffer.writeln('    Allowed values: ${param.enumValues!.join(', ')}');
        }
      }
      buffer.writeln();
    }

    buffer.writeln('To call a function, respond with a JSON object in this format:');
    buffer.writeln('```json');
    buffer.writeln('{');
    buffer.writeln('  "name": "function_name",');
    buffer.writeln('  "arguments": {');
    buffer.writeln('    "param1": "value1",');
    buffer.writeln('    "param2": "value2"');
    buffer.writeln('  }');
    buffer.writeln('}');
    buffer.writeln('```');
    buffer.writeln();
    buffer.writeln('Only call functions when necessary. If no function call is needed, respond normally.');

    return buffer.toString();
  }

  /// 生成 Llama 3 Instruct 格式的函数提示
  static String generateLlama3Prompt(List<FunctionDefinition> functions) {
    final toolsJson = functions.map((f) => f.toOpenAiSchema()).toList();
    return '''
Available tools:
${jsonEncode(toolsJson)}

When you need to call a function, output the function call in the following format:
<|start_header_id|>assistant<|end_header_id|>

{"name": "function_name", "arguments": {"arg1": "value1"}}
''';
  }

  /// 生成 Qwen 格式的函数提示
  static String generateQwenPrompt(List<FunctionDefinition> functions) {
    final toolsJson = functions.map((f) => f.toOpenAiSchema()).toList();
    return '''
# Tools

You may call one or more functions to assist with the user query.
You are provided with function descriptions in the following format:

${_jsonEncode(toolsJson)}

For each function call, output the function name and arguments within the following XML tags:
<tool_call name="function_name">
{"arg1": "value1"}
</tool_call >
''';
  }
}

String _jsonEncode(dynamic obj) {
  return JsonEncoder.withIndent('  ').convert(obj);
}
