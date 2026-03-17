import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:llama_native/src/llama_native_bindings.dart' as bindings;
import 'package:llama_native/llama_chat_message.dart';
import 'package:llama_native/src/log/logger.dart';

/// Chat Template 类型
enum LlamaTemplateType {
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

/// Chat Template 应用器
class LlamaTemplate {
  final Logger _logger = Logger('LlamaTemplate');

  /// 应用原生模板（使用 llama_chat_apply_template）
  String applyNative(List<LlamaChatMessage> messages, {String? template}) {
    try {
      final chatMessages = _toNativeChatMessages(messages);

      try {
        final totalChars = messages.fold<int>(0, (sum, msg) => sum + msg.content.length);
        var bufferSize = (totalChars * 4).clamp(1024, 1024 * 1024);
        var buffer = calloc<Char>(bufferSize);

        try {
          var result = bindings.llama_chat_apply_template(
            nullptr,
            chatMessages,
            messages.length,
            true,
            buffer,
            bufferSize,
          );

          if (result > bufferSize) {
            calloc.free(buffer);
            bufferSize = result + 1;
            buffer = calloc<Char>(bufferSize);

            result = bindings.llama_chat_apply_template(
              nullptr,
              chatMessages,
              messages.length,
              true,
              buffer,
              bufferSize,
            );
          }

          if (result > 0) {
            final formattedText = buffer.cast<Utf8>().toDartString(length: result);
            _logger.debug('Applied native template, length=$result');
            return formattedText;
          } else {
            _logger.error('Failed to apply template, result: $result');
            return applyFallback(messages, LlamaTemplateType.chatml);
          }
        } finally {
          calloc.free(buffer);
        }
      } finally {
        _freeChatMessages(chatMessages, messages.length);
      }
    } catch (e) {
      _logger.error('Native template application failed: $e');
      return applyFallback(messages, LlamaTemplateType.chatml);
    }
  }

  /// 应用回退模板
  String applyFallback(List<LlamaChatMessage> messages, LlamaTemplateType type) {
    switch (type) {
      case LlamaTemplateType.llama3:
        return _applyLlama3Template(messages);
      case LlamaTemplateType.qwen:
        return _applyQwenTemplate(messages);
      case LlamaTemplateType.mistral:
        return _applyMistralTemplate(messages);
      case LlamaTemplateType.chatml:
        return _applyChatMLTemplate(messages);
      case LlamaTemplateType.alpaca:
        return _applyAlpacaTemplate(messages);
      case LlamaTemplateType.unknown:
        return _applyDefaultTemplate(messages);
    }
  }

  /// 根据模型名称检测模板类型
  LlamaTemplateType detectType(String modelName) {
    final name = modelName.toLowerCase();

    if (name.contains('llama-3') || name.contains('llama3')) {
      return LlamaTemplateType.llama3;
    } else if (name.contains('qwen')) {
      return LlamaTemplateType.qwen;
    } else if (name.contains('mistral')) {
      return LlamaTemplateType.mistral;
    } else if (name.contains('chatml')) {
      return LlamaTemplateType.chatml;
    } else if (name.contains('alpaca')) {
      return LlamaTemplateType.alpaca;
    }

    return LlamaTemplateType.unknown;
  }

  /// 转换为原生 llama_chat_message 数组
  Pointer<bindings.llama_chat_message> _toNativeChatMessages(List<LlamaChatMessage> messages) {
    final chatMessages = calloc<bindings.llama_chat_message>(messages.length);

    for (var i = 0; i < messages.length; i++) {
      final msg = messages[i];
      final chatMsg = chatMessages.elementAt(i).ref;
      chatMsg.role = msg.role.name.toNativeUtf8().cast<Char>();
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

  String _applyLlama3Template(List<LlamaChatMessage> messages) {
    final buffer = StringBuffer();

    for (final msg in messages) {
      final role = msg.role.name;
      buffer.write('<|start_header_id|>$role<|end_header_id|>\n\n');
      buffer.write(msg.content.trim());
      buffer.write('<|eot_id|>');
    }

    buffer.write('<|start_header_id|>assistant<|end_header_id|>\n\n');

    return buffer.toString();
  }

  String _applyQwenTemplate(List<LlamaChatMessage> messages) {
    final buffer = StringBuffer();

    for (final msg in messages) {
      final role = msg.role.name;
      buffer.write('<|im_start|>$role\n');
      buffer.write(msg.content.trim());
      buffer.write('<|im_end|>');
    }

    buffer.write('<|im_start|>assistant\n');

    return buffer.toString();
  }

  String _applyMistralTemplate(List<LlamaChatMessage> messages) {
    final buffer = StringBuffer();

    for (final msg in messages) {
      if (msg.role == LlamaChatMessageRole.user) {
        buffer.write('[INST] ${msg.content} [/INST]');
      } else if (msg.role == LlamaChatMessageRole.assistant) {
        buffer.write(' ${msg.content}');
      } else {
        buffer.write('${msg.content}\n');
      }
    }

    return buffer.toString();
  }

  String _applyChatMLTemplate(List<LlamaChatMessage> messages) {
    final buffer = StringBuffer();

    for (final msg in messages) {
      final role = msg.role.name;
      buffer.write('<|im_start|>$role\n');
      buffer.write(msg.content.trim());
      buffer.write('<|im_end|>\n');
    }

    buffer.write('<|im_start|>assistant\n');

    return buffer.toString();
  }

  String _applyAlpacaTemplate(List<LlamaChatMessage> messages) {
    final buffer = StringBuffer();

    buffer.write(
      'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n',
    );

    for (final msg in messages) {
      if (msg.role == LlamaChatMessageRole.user) {
        buffer.write('### Instruction:\n${msg.content}\n\n');
      } else if (msg.role == LlamaChatMessageRole.assistant) {
        buffer.write('### Response:\n${msg.content}\n\n');
      } else {
        buffer.write('${msg.content}\n\n');
      }
    }

    buffer.write('### Response:\n');

    return buffer.toString();
  }

  String _applyDefaultTemplate(List<LlamaChatMessage> messages) {
    final buffer = StringBuffer();

    for (final msg in messages) {
      buffer.write(msg.content);
      buffer.write('\n');
    }

    return buffer.toString();
  }
}
