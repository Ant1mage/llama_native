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
  const ChatMessage.system(String content) : role = MessageRole.system, content = content;

  /// 创建用户消息
  const ChatMessage.user(String content) : role = MessageRole.user, content = content;

  /// 创建助手消息
  const ChatMessage.assistant(String content) : role = MessageRole.assistant, content = content;
}

/// 协议适配器
///
/// 负责：
/// - 自动识别 Chat Template (Llama3/Qwen/Mistral)
/// - 应用特殊 Token 解析
/// - 配置 add_bos, parse_special_tokens
class ChatTokenizer {
  final LlamaModel _model;
  final Logger _logger;

  ChatTemplateType _templateType;
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
       _templateType = templateType ?? ChatTemplateType.unknown,
       _addBos = addBos,
       _addEos = addEos,
       _parseSpecialTokens = parseSpecialTokens,
       _logger = Logger('ChatTokenizer') {
    // 自动检测模板类型
    if (_templateType == ChatTemplateType.unknown) {
      _templateType = _detectTemplateType();
      _logger.info('Auto-detected template type: $_templateType');
    }
  }

  /// BOS token
  bool get addBos => _addBos;

  /// EOS token
  bool get addEos => _addEos;

  /// 解析特殊 token
  bool get parseSpecialTokens => _parseSpecialTokens;

  /// 检测模板类型
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
    switch (_templateType) {
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
      final role = msg.role == MessageRole.user ? 'user' : 'assistant';
      buffer.write('\n\n$role:$msg.content');
    }

    // 添加助手前缀
    buffer.write('\n\nassistant:');

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
