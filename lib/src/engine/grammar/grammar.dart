import 'dart:convert';
import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:llama_native/src/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/engine/grammar/json_schema_to_gbnf.dart';
import 'package:llama_native/src/log/logger.dart';

class GrammarConfig {
  final String grammarStr;
  final String rootRule;
  final bool lazy;
  final List<String> triggerWords;
  final List<int> triggerTokens;

  const GrammarConfig({
    required this.grammarStr,
    this.rootRule = 'root',
    this.lazy = false,
    this.triggerWords = const [],
    this.triggerTokens = const [],
  });

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

class GrammarParseResult {
  final String rawString;

  GrammarParseResult(this.rawString);

  Map<String, dynamic> toMap() {
    try {
      final decoded = jsonDecode(rawString);
      return decoded as Map<String, dynamic>;
    } catch (e) {
      throw FormatException('解析Grammar结果为JSON失败: $e');
    }
  }

  List<dynamic> toList() {
    try {
      final decoded = jsonDecode(rawString);
      return decoded as List<dynamic>;
    } catch (e) {
      throw FormatException('解析Grammar结果为JSON失败: $e');
    }
  }

  dynamic toDynamic() {
    try {
      return jsonDecode(rawString);
    } catch (e) {
      throw FormatException('解析Grammar结果为JSON失败: $e');
    }
  }

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

class Grammar {
  final GrammarConfig _config;
  final Pointer<bindings.llama_vocab> _vocab;
  final Logger _logger;

  Pointer<bindings.llama_sampler>? _sampler;

  Grammar._(this._config, this._vocab) : _logger = Logger('Grammar');

  factory Grammar.create(GrammarConfig config, Pointer<bindings.llama_vocab> vocab) {
    final grammar = Grammar._(config, vocab);
    grammar._initialize();
    return grammar;
  }

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

  void _initialize() {
    _logger.debug('初始化Grammar，根规则: ${_config.rootRule}');

    final grammarStrC = _config.grammarStr.toNativeUtf8().cast<Char>();
    final rootRuleC = _config.rootRule.toNativeUtf8().cast<Char>();

    try {
      if (_config.lazy && (_config.triggerWords.isNotEmpty || _config.triggerTokens.isNotEmpty)) {
        _sampler = _createLazySampler(grammarStrC, rootRuleC);
      } else {
        _sampler = bindings.llama_sampler_init_grammar(_vocab, grammarStrC, rootRuleC);
      }

      if (_sampler == null || _sampler == nullptr) {
        throw StateError('创建Grammar采样器失败');
      }

      _logger.info('Grammar采样器初始化成功');
    } catch (e) {
      _logger.error('初始化Grammar失败: $e');
      rethrow;
    } finally {
      calloc.free(grammarStrC);
      calloc.free(rootRuleC);
    }
  }

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

  Pointer<bindings.llama_sampler> get sampler {
    if (_sampler == null) {
      throw StateError('Grammar is disposed');
    }
    return _sampler!;
  }

  String get grammarString => _config.grammarStr;
  String get rootRule => _config.rootRule;

  GrammarParseResult parse(String text) {
    return GrammarParseResult(text);
  }

  void reset() {
    if (_sampler != null && _sampler != nullptr) {
      bindings.llama_sampler_reset(_sampler!);
      _logger.debug('Grammar采样器已重置');
    }
  }

  void dispose() {
    if (_sampler == null) return;

    _logger.debug('释放Grammar...');

    bindings.llama_sampler_free(_sampler!);
    _sampler = null;

    _logger.debug('Grammar已释放');
  }
}

class JsonSchemaBuilder {
  final Map<String, dynamic> _schema = {};

  JsonSchemaBuilder();

  JsonSchemaBuilder type(String type) {
    _schema['type'] = type;
    return this;
  }

  JsonSchemaBuilder property(String name, Map<String, dynamic> propertySchema) {
    _schema['properties'] ??= <String, dynamic>{};
    (_schema['properties'] as Map<String, dynamic>)[name] = propertySchema;
    return this;
  }

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

  JsonSchemaBuilder booleanProperty(String name, {String? description}) {
    final prop = <String, dynamic>{'type': 'boolean'};
    if (description != null) prop['description'] = description;
    return property(name, prop);
  }

  JsonSchemaBuilder arrayProperty(String name, {Map<String, dynamic>? items, int? minItems, int? maxItems}) {
    final prop = <String, dynamic>{'type': 'array'};
    if (items != null) prop['items'] = items;
    if (minItems != null) prop['minItems'] = minItems;
    if (maxItems != null) prop['maxItems'] = maxItems;
    return property(name, prop);
  }

  JsonSchemaBuilder objectProperty(String name, {Map<String, dynamic>? properties, List<String>? required}) {
    final prop = <String, dynamic>{'type': 'object'};
    if (properties != null) prop['properties'] = properties;
    if (required != null) prop['required'] = required;
    return property(name, prop);
  }

  JsonSchemaBuilder required(List<String> properties) {
    _schema['required'] = properties;
    return this;
  }

  JsonSchemaBuilder enumValues(List<dynamic> values) {
    _schema['enum'] = values;
    return this;
  }

  JsonSchemaBuilder description(String desc) {
    _schema['description'] = desc;
    return this;
  }

  JsonSchemaBuilder oneOf(List<Map<String, dynamic>> schemas) {
    _schema['oneOf'] = schemas;
    return this;
  }

  JsonSchemaBuilder anyOf(List<Map<String, dynamic>> schemas) {
    _schema['anyOf'] = schemas;
    return this;
  }

  Map<String, dynamic> build() {
    return Map.unmodifiable(_schema);
  }
}
