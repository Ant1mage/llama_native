import 'dart:convert';
import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:llama_native/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/grammar/json_schema_to_gbnf.dart';
import 'package:llama_native/src/logging/logger.dart';
import 'package:llama_native/src/utils/disposable.dart';

/// Grammar 配置
class GrammarConfig {
  /// GBNF grammar 字符串
  final String grammarStr;

  /// 根规则名称
  final String rootRule;

  /// 是否使用懒加载模式
  final bool lazy;

  /// 触发词列表（懒加载模式）
  final List<String> triggerWords;

  /// 触发 token 列表（懒加载模式）
  final List<int> triggerTokens;

  const GrammarConfig({
    required this.grammarStr,
    this.rootRule = 'root',
    this.lazy = false,
    this.triggerWords = const [],
    this.triggerTokens = const [],
  });

  /// 从 JSON Schema 创建配置
  factory GrammarConfig.fromJsonSchema(
    Map<String, dynamic> schema, {
    String rootRule = 'root',
    bool lazy = false,
    List<String> triggerWords = const [],
    List<int> triggerTokens = const [],
  }) {
    final grammarStr = generateGbnfGrammar(schema, rootName: rootRule);
    return GrammarConfig(
      grammarStr: grammarStr,
      rootRule: rootRule,
      lazy: lazy,
      triggerWords: triggerWords,
      triggerTokens: triggerTokens,
    );
  }
}

/// Grammar 解析结果
class GrammarParseResult {
  /// 原始字符串
  final String rawString;

  GrammarParseResult(this.rawString);

  /// 解析为 Map
  Map<String, dynamic> toMap() {
    try {
      final decoded = jsonDecode(rawString);
      return decoded as Map<String, dynamic>;
    } catch (e) {
      throw FormatException('Failed to parse grammar result as JSON: $e');
    }
  }

  /// 解析为 List
  List<dynamic> toList() {
    try {
      final decoded = jsonDecode(rawString);
      return decoded as List<dynamic>;
    } catch (e) {
      throw FormatException('Failed to parse grammar result as JSON: $e');
    }
  }

  /// 解析为动态类型
  dynamic toDynamic() {
    try {
      return jsonDecode(rawString);
    } catch (e) {
      throw FormatException('Failed to parse grammar result as JSON: $e');
    }
  }

  /// 尝试解析，失败返回 null
  Map<String, dynamic>? tryToMap() {
    try {
      return toMap();
    } catch (_) {
      return null;
    }
  }

  @override
  String toString() => rawString;
}

/// GBNF Grammar 包装器
///
/// 提供：
/// - JSON Schema 到 GBNF 的转换
/// - Grammar sampler 的创建和管理
/// - 结果解析
class Grammar with Disposable {
  final GrammarConfig _config;
  final Pointer<bindings.llama_vocab> _vocab;
  final Logger _logger;

  Pointer<bindings.llama_sampler>? _sampler;
  bool _disposed = false;

  Grammar._(this._config, this._vocab) : _logger = Logger('Grammar');

  /// 从配置创建 Grammar
  factory Grammar.create(GrammarConfig config, Pointer<bindings.llama_vocab> vocab) {
    final grammar = Grammar._(config, vocab);
    grammar._initialize();
    return grammar;
  }

  /// 从 JSON Schema 创建 Grammar
  factory Grammar.fromJsonSchema(
    Map<String, dynamic> schema,
    Pointer<bindings.llama_vocab> vocab, {
    String rootRule = 'root',
    bool lazy = false,
    List<String> triggerWords = const [],
    List<int> triggerTokens = const [],
  }) {
    final config = GrammarConfig.fromJsonSchema(
      schema,
      rootRule: rootRule,
      lazy: lazy,
      triggerWords: triggerWords,
      triggerTokens: triggerTokens,
    );
    return Grammar.create(config, vocab);
  }

  /// 初始化 grammar sampler
  void _initialize() {
    _logger.debug('Initializing grammar with root rule: ${_config.rootRule}');

    final grammarStrC = _config.grammarStr.toNativeUtf8().cast<Char>();
    final rootRuleC = _config.rootRule.toNativeUtf8().cast<Char>();

    try {
      if (_config.lazy && (_config.triggerWords.isNotEmpty || _config.triggerTokens.isNotEmpty)) {
        _sampler = _createLazySampler(grammarStrC, rootRuleC);
      } else {
        _sampler = bindings.llama_sampler_init_grammar(_vocab, grammarStrC, rootRuleC);
      }

      if (_sampler == null || _sampler == nullptr) {
        throw StateError('Failed to create grammar sampler');
      }

      _logger.info('Grammar sampler initialized successfully');
    } catch (e) {
      _logger.error('Failed to initialize grammar: $e');
      rethrow;
    } finally {
      calloc.free(grammarStrC);
      calloc.free(rootRuleC);
    }
  }

  /// 创建懒加载 grammar sampler
  Pointer<bindings.llama_sampler> _createLazySampler(Pointer<Char> grammarStrC, Pointer<Char> rootRuleC) {
    Pointer<Pointer<Char>>? triggerWordsPtr;
    Pointer<bindings.llama_token>? triggerTokensPtr;

    try {
      if (_config.triggerWords.isNotEmpty) {
        triggerWordsPtr = calloc<Pointer<Char>>(_config.triggerWords.length);
        for (var i = 0; i < _config.triggerWords.length; i++) {
          triggerWordsPtr[i] = _config.triggerWords[i].toNativeUtf8().cast<Char>();
        }
      }

      if (_config.triggerTokens.isNotEmpty) {
        triggerTokensPtr = calloc<bindings.llama_token>(_config.triggerTokens.length);
        for (var i = 0; i < _config.triggerTokens.length; i++) {
          triggerTokensPtr[i] = _config.triggerTokens[i];
        }
      }

      return bindings.llama_sampler_init_grammar_lazy_patterns(
        _vocab,
        grammarStrC,
        rootRuleC,
        triggerWordsPtr ?? nullptr,
        _config.triggerWords.length,
        triggerTokensPtr ?? nullptr,
        _config.triggerTokens.length,
      );
    } finally {
      if (triggerWordsPtr != null) {
        for (var i = 0; i < _config.triggerWords.length; i++) {
          calloc.free(triggerWordsPtr[i]);
        }
        calloc.free(triggerWordsPtr);
      }
      if (triggerTokensPtr != null) {
        calloc.free(triggerTokensPtr);
      }
    }
  }

  /// 获取 sampler 指针
  Pointer<bindings.llama_sampler> get sampler {
    if (_disposed || _sampler == null) {
      throw StateError('Grammar is disposed');
    }
    return _sampler!;
  }

  /// 获取 grammar 字符串
  String get grammarString => _config.grammarStr;

  /// 获取根规则名称
  String get rootRule => _config.rootRule;

  /// 解析生成的文本
  GrammarParseResult parse(String text) {
    return GrammarParseResult(text);
  }

  /// 重置 grammar 状态
  void reset() {
    if (_sampler != null && _sampler != nullptr) {
      bindings.llama_sampler_reset(_sampler!);
      _logger.debug('Grammar sampler reset');
    }
  }

  @override
  bool get isDisposed => _disposed;

  @override
  void dispose() {
    if (_disposed) return;

    _logger.debug('Disposing grammar...');

    if (_sampler != null && _sampler != nullptr) {
      bindings.llama_sampler_free(_sampler!);
      _sampler = null;
    }

    _disposed = true;
    _logger.debug('Grammar disposed');
  }
}

/// JSON Schema 构建器
///
/// 提供流畅的 API 来构建 JSON Schema
class JsonSchemaBuilder {
  final Map<String, dynamic> _schema = {};

  JsonSchemaBuilder();

  /// 设置类型
  JsonSchemaBuilder type(String type) {
    _schema['type'] = type;
    return this;
  }

  /// 添加属性
  JsonSchemaBuilder property(String name, Map<String, dynamic> propertySchema) {
    _schema['properties'] ??= <String, dynamic>{};
    (_schema['properties'] as Map<String, dynamic>)[name] = propertySchema;
    return this;
  }

  /// 添加字符串属性
  JsonSchemaBuilder stringProperty(
    String name, {
    String? description,
    int? minLength,
    int? maxLength,
    String? pattern,
    List<String>? enumValues,
    String? constValue,
  }) {
    final prop = <String, dynamic>{'type': 'string'};
    if (description != null) prop['description'] = description;
    if (minLength != null) prop['minLength'] = minLength;
    if (maxLength != null) prop['maxLength'] = maxLength;
    if (pattern != null) prop['pattern'] = pattern;
    if (enumValues != null) prop['enum'] = enumValues;
    if (constValue != null) prop['const'] = constValue;
    return property(name, prop);
  }

  /// 添加数字属性
  JsonSchemaBuilder numberProperty(
    String name, {
    String? description,
    double? minimum,
    double? maximum,
    bool isInteger = false,
  }) {
    final prop = <String, dynamic>{'type': isInteger ? 'integer' : 'number'};
    if (description != null) prop['description'] = description;
    if (minimum != null) prop['minimum'] = minimum;
    if (maximum != null) prop['maximum'] = maximum;
    return property(name, prop);
  }

  /// 添加布尔属性
  JsonSchemaBuilder booleanProperty(String name, {String? description}) {
    final prop = <String, dynamic>{'type': 'boolean'};
    if (description != null) prop['description'] = description;
    return property(name, prop);
  }

  /// 添加数组属性
  JsonSchemaBuilder arrayProperty(String name, {Map<String, dynamic>? items, int? minItems, int? maxItems}) {
    final prop = <String, dynamic>{'type': 'array'};
    if (items != null) prop['items'] = items;
    if (minItems != null) prop['minItems'] = minItems;
    if (maxItems != null) prop['maxItems'] = maxItems;
    return property(name, prop);
  }

  /// 添加对象属性
  JsonSchemaBuilder objectProperty(String name, {Map<String, dynamic>? properties, List<String>? required}) {
    final prop = <String, dynamic>{'type': 'object'};
    if (properties != null) prop['properties'] = properties;
    if (required != null) prop['required'] = required;
    return property(name, prop);
  }

  /// 设置必需属性
  JsonSchemaBuilder required(List<String> properties) {
    _schema['required'] = properties;
    return this;
  }

  /// 添加枚举约束
  JsonSchemaBuilder enumValues(List<dynamic> values) {
    _schema['enum'] = values;
    return this;
  }

  /// 设置描述
  JsonSchemaBuilder description(String desc) {
    _schema['description'] = desc;
    return this;
  }

  /// 添加 oneOf 约束
  JsonSchemaBuilder oneOf(List<Map<String, dynamic>> schemas) {
    _schema['oneOf'] = schemas;
    return this;
  }

  /// 添加 anyOf 约束
  JsonSchemaBuilder anyOf(List<Map<String, dynamic>> schemas) {
    _schema['anyOf'] = schemas;
    return this;
  }

  /// 构建最终的 schema
  Map<String, dynamic> build() {
    return Map.unmodifiable(_schema);
  }
}
