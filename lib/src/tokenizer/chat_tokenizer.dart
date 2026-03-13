import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:llama_native/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/model/llama_model.dart';
import 'package:llama_native/src/logging/logger.dart';

/// Chat Template 类型
enum ChatTemplateType {
  /// Llama3 格式
  llama3,

  /// Qwen 格式
  qwen,

  /// Mistral 格式
  mistral,

  /// ChatML 格式
  chatml,

  /// Alpaca 格式
  alpaca,

  /// 未知格式 (使用默认)
  unknown,
}

/// 消息角色
enum MessageRole {
  /// 系统提示
  system,

  /// 用户消息
  user,

  /// 助手回复
  assistant,
}

/// Chat 消息
class ChatMessage {
  final MessageRole role;
  final String content;

  const ChatMessage({required this.role, required this.content});

  /// 创建系统消息
  const ChatMessage.system(this.content) : role = MessageRole.system;

  /// 创建用户消息
  const ChatMessage.user(this.content) : role = MessageRole.user;

  /// 创建助手消息
  const ChatMessage.assistant(this.content) : role = MessageRole.assistant;
}

/// 协议适配器
///
/// 负责：
/// - 优先使用模型内置的 chat template（如果存在）
/// - 回退到内置模板（Llama3/Qwen/Mistral 等）
/// - 应用特殊 Token 解析
/// - 配置 add_bos, parse_special_tokens
class ChatTokenizer {
  final LlamaModel _model;
  final Logger _logger;

  String? _modelTemplate; // 从模型获取的 template
  ChatTemplateType _fallbackTemplateType; // 回退模板类型
  bool _addBos;
  bool _addEos;
  bool _parseSpecialTokens;

  /// 创建 ChatTokenizer
  ChatTokenizer({
    required LlamaModel model,
    ChatTemplateType? templateType,
    bool addBos = true,
    bool addEos = false,
    bool parseSpecialTokens = true,
  }) : _model = model,
       _fallbackTemplateType = templateType ?? ChatTemplateType.unknown,
       _addBos = addBos,
       _addEos = addEos,
       _parseSpecialTokens = parseSpecialTokens,
       _logger = Logger('ChatTokenizer') {
    // 尝试从模型获取内置 template
    _modelTemplate = _getModelTemplate();

    if (_modelTemplate == null) {
      _logger.info('No built-in template found in model, using fallback');
      // 如果没有内置 template，使用回退方案
      if (_fallbackTemplateType == ChatTemplateType.unknown) {
        _fallbackTemplateType = _detectTemplateType();
        _logger.info('Auto-detected fallback template type: $_fallbackTemplateType');
      }
    } else {
      _logger.info('Using built-in template from model');
    }
  }

  /// 从模型获取内置 template
  String? _getModelTemplate() {
    try {
      // 使用 nullptr 获取默认 template
      final templatePtr = bindings.llama_model_chat_template(_model.handle, nullptr);

      if (templatePtr == nullptr) {
        return null;
      }

      final template = templatePtr.cast<Utf8>().toDartString();
      _logger.debug('Found model template: ${template.substring(0, template.length.clamp(0, 100))}...');
      return template;
    } catch (e) {
      _logger.warning('Failed to get model template: $e');
      return null;
    }
  }

  /// BOS token
  bool get addBos => _addBos;

  /// EOS token
  bool get addEos => _addEos;

  /// 解析特殊 token
  bool get parseSpecialTokens => _parseSpecialTokens;

  /// 检测模板类型（回退方案）
  ChatTemplateType _detectTemplateType() {
    // 从模型元数据推断
    final metadata = _model.getMetadata();
    final name = metadata.name.toLowerCase();

    if (name.contains('llama-3') || name.contains('llama3')) {
      return ChatTemplateType.llama3;
    } else if (name.contains('qwen')) {
      return ChatTemplateType.qwen;
    } else if (name.contains('mistral')) {
      return ChatTemplateType.mistral;
    } else if (name.contains('chatml')) {
      return ChatTemplateType.chatml;
    } else if (name.contains('alpaca')) {
      return ChatTemplateType.alpaca;
    }

    _logger.warning('Unknown template type, using default');
    return ChatTemplateType.unknown;
  }

  /// 应用 Chat Template
  String applyTemplate(List<ChatMessage> messages) {
    // 优先使用模型内置 template
    if (_modelTemplate != null) {
      return _applyNativeTemplate(messages);
    }

    // 回退到内置模板
    return _applyFallbackTemplate(messages);
  }

  /// 使用原生 API 应用模板
  String _applyNativeTemplate(List<ChatMessage> messages) {
    try {
      // 转换为 llama_chat_message 数组
      final chatMessages = _toNativeChatMessages(messages);

      try {
        // 估算缓冲区大小（推荐 2 * 总字符数）
        final totalChars = messages.fold<int>(0, (sum, msg) => sum + msg.content.length);
        final bufferSize = (totalChars * 2).clamp(1024, 1024 * 1024);
        final buffer = calloc<Char>(bufferSize);

        try {
          final result = bindings.llama_chat_apply_template(
            nullptr, // 使用模型默认 template
            chatMessages,
            messages.length,
            true, // add_ass
            buffer,
            bufferSize,
          );

          if (result > 0 && result <= bufferSize) {
            final formattedText = buffer.cast<Utf8>().toDartString(length: result);
            _logger.debug('Applied native template: $formattedText');
            return formattedText;
          } else if (result > bufferSize) {
            _logger.warning('Buffer too small, need $result bytes');
            // 重新分配更大的缓冲区
            calloc.free(buffer);
            final newBuffer = calloc<Char>(result);
            try {
              bindings.llama_chat_apply_template(nullptr, chatMessages, messages.length, true, newBuffer, result);
              return newBuffer.cast<Utf8>().toDartString(length: result);
            } finally {
              calloc.free(newBuffer);
            }
          } else {
            _logger.error('Failed to apply template, result: $result');
            return _applyFallbackTemplate(messages);
          }
        } finally {
          calloc.free(buffer);
        }
      } finally {
        _freeChatMessages(chatMessages, messages.length);
      }
    } catch (e) {
      _logger.error('Native template application failed: $e');
      return _applyFallbackTemplate(messages);
    }
  }

  /// 转换为原生 llama_chat_message 数组
  Pointer<bindings.llama_chat_message> _toNativeChatMessages(List<ChatMessage> messages) {
    final chatMessages = calloc<bindings.llama_chat_message>(messages.length);

    for (var i = 0; i < messages.length; i++) {
      final msg = messages[i];
      final chatMsg = chatMessages.elementAt(i).ref;

      // 设置 role
      chatMsg.role = msg.role.name.toNativeUtf8().cast<Char>();

      // 设置 content
      chatMsg.content = msg.content.toNativeUtf8().cast<Char>();
    }

    return chatMessages;
  }

  /// 释放聊天消息内存
  void _freeChatMessages(Pointer<bindings.llama_chat_message> messages, int length) {
    for (var i = 0; i < length; i++) {
      final msg = messages.elementAt(i).ref;
      calloc.free(msg.role);
      calloc.free(msg.content);
    }
    calloc.free(messages);
  }

  /// 回退到内置模板
  String _applyFallbackTemplate(List<ChatMessage> messages) {
    switch (_fallbackTemplateType) {
      case ChatTemplateType.llama3:
        return _applyLlama3Template(messages);
      case ChatTemplateType.qwen:
        return _applyQwenTemplate(messages);
      case ChatTemplateType.mistral:
        return _applyMistralTemplate(messages);
      case ChatTemplateType.chatml:
        return _applyChatMLTemplate(messages);
      case ChatTemplateType.alpaca:
        return _applyAlpacaTemplate(messages);
      case ChatTemplateType.unknown:
        return _applyDefaultTemplate(messages);
    }
  }

  /// Llama3 模板
  String _applyLlama3Template(List<ChatMessage> messages) {
    final buffer = StringBuffer();

    for (final msg in messages) {
      final role = msg.role.name;
      buffer.write('<|start_header_id|>$role<|end_header_id|>\n\n');
      buffer.write(msg.content.trim());
      buffer.write('<|eot_id|>');
    }

    // 添加助手前缀
    buffer.write('<|start_header_id|>assistant<|end_header_id|>\n\n');

    return buffer.toString();
  }

  /// Qwen 模板
  String _applyQwenTemplate(List<ChatMessage> messages) {
    final buffer = StringBuffer();

    for (final msg in messages) {
      final role = msg.role.name;
      buffer.write('<|im_start|>$role\n');
      buffer.write(msg.content.trim());
      buffer.write('<|im_end|>');
    }

    // 添加助手前缀
    buffer.write('<|im_start|>assistant\n');

    return buffer.toString();
  }

  /// Mistral 模板
  String _applyMistralTemplate(List<ChatMessage> messages) {
    final buffer = StringBuffer();

    for (final msg in messages) {
      if (msg.role == MessageRole.user) {
        buffer.write('[INST] ${msg.content} [/INST]');
      } else if (msg.role == MessageRole.assistant) {
        buffer.write(' ${msg.content}');
      } else {
        buffer.write('${msg.content}\n');
      }
    }

    return buffer.toString();
  }

  /// ChatML 模板
  String _applyChatMLTemplate(List<ChatMessage> messages) {
    final buffer = StringBuffer();

    for (final msg in messages) {
      final role = msg.role.name;
      buffer.write('<|im_start|>$role\n');
      buffer.write(msg.content.trim());
      buffer.write('<|im_end|>\n');
    }

    // 添加助手前缀
    buffer.write('<|im_start|>assistant\n');

    return buffer.toString();
  }

  /// Alpaca 模板
  String _applyAlpacaTemplate(List<ChatMessage> messages) {
    final buffer = StringBuffer();

    buffer.write(
      'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n',
    );

    for (final msg in messages) {
      if (msg.role == MessageRole.user) {
        buffer.write('### Instruction:\n${msg.content}\n\n');
      } else if (msg.role == MessageRole.assistant) {
        buffer.write('### Response:\n${msg.content}\n\n');
      } else {
        buffer.write('${msg.content}\n\n');
      }
    }

    buffer.write('### Response:\n');

    return buffer.toString();
  }

  /// 默认模板 (简单拼接)
  String _applyDefaultTemplate(List<ChatMessage> messages) {
    final buffer = StringBuffer();

    for (final msg in messages) {
      buffer.write(msg.content);
      buffer.write('\n');
    }

    return buffer.toString();
  }

  /// Tokenize 对话
  List<int> tokenizeMessages(List<ChatMessage> messages) {
    final formattedText = applyTemplate(messages);
    return _model.tokenize(formattedText, addBos: _addBos, addEos: _addEos);
  }
}
