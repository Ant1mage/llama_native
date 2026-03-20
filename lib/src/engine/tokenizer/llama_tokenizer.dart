import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:llama_native/src/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/engine/model/llama_model.dart';
import 'package:llama_native/src/log/logger.dart';
import 'package:llama_native/llama_chat_message.dart';
import 'package:llama_native/src/engine/tokenizer/llama_template.dart';

class LlamaTokenizer {
  final LlamaModel _model;
  final Logger _logger;
  final LlamaTemplate _template = LlamaTemplate();

  String? _modelTemplate;
  bool _addBos;
  bool _addEos;

  LlamaTokenizer({required LlamaModel model, bool addBos = true, bool addEos = false})
    : _model = model,
      _addBos = addBos,
      _addEos = addEos,
      _logger = Logger('LlamaTokenizer') {
    _modelTemplate = _getModelTemplate();
    if (_modelTemplate != null) {
      _logger.info('使用模型内置模板');
    } else {
      _logger.info('未找到内置模板，将使用原生API');
    }
  }

  String? _getModelTemplate() {
    try {
      final templatePtr = bindings.llama_model_chat_template(_model.handle, nullptr);
      if (templatePtr == nullptr) return null;
      final template = templatePtr.cast<Utf8>().toDartString();
      _logger.debug('找到模型模板: ${template.substring(0, template.length.clamp(0, 100))}...');
      return template;
    } catch (e) {
      _logger.warning('获取模型模板失败: $e');
      return null;
    }
  }

  String applyTemplate(List<LlamaChatMessage> messages) {
    return _template.applyNative(messages, template: _modelTemplate);
  }

  List<int> tokenizeMessages(List<LlamaChatMessage> messages) {
    final formattedText = applyTemplate(messages);
    return _model.tokenize(formattedText, addBos: _addBos, addEos: _addEos);
  }

  bool get addBos => _addBos;
  bool get addEos => _addEos;
}
