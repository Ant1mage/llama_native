import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:llama_native/src/llama_native_bindings.dart' as bindings;
import 'package:llama_native/llama_chat_message.dart';
import 'package:llama_native/src/log/logger.dart';

enum LlamaTemplateType {
  llama3,
  qwen,
  mistral,
  chatml,
  alpaca,
  gemma,
  minicpm,
  unknown,
}

class LlamaTemplate {
  final Logger _logger = Logger('LlamaTemplate');

  String applyNative(List<LlamaChatMessage> messages, {String? template}) {
    Pointer<bindings.llama_chat_message>? chatMessages;
    Pointer<Char>? tmplPtr;

    try {
      chatMessages = _toNativeChatMessages(messages);

      final tmpl = template;
      if (tmpl != null) {
        tmplPtr = tmpl.toNativeUtf8().cast<Char>();
      }

      final totalChars = messages.fold<int>(0, (sum, msg) => sum + msg.content.length);
      var bufferSize = (totalChars * 4).clamp(1024, 1024 * 1024);
      var buffer = calloc<Char>(bufferSize);

      try {
        var result = bindings.llama_chat_apply_template(
          tmplPtr ?? nullptr,
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
            tmplPtr ?? nullptr,
            chatMessages,
            messages.length,
            true,
            buffer,
            bufferSize,
          );
        }

        if (result > 0) {
          final formattedText = buffer.cast<Utf8>().toDartString(length: result);
          _logger.debug('应用原生模板，长度=$result');
          return formattedText;
        } else {
          _logger.error('应用模板失败，结果: $result');
          return applyFallback(messages, LlamaTemplateType.chatml);
        }
      } finally {
        calloc.free(buffer);
      }
    } catch (e) {
      _logger.error('原生模板应用失败: $e');
      return applyFallback(messages, LlamaTemplateType.chatml);
    } finally {
      if (chatMessages != null) {
        _freeChatMessages(chatMessages, messages.length);
      }
      if (tmplPtr != null) {
        calloc.free(tmplPtr);
      }
    }
  }

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
      case LlamaTemplateType.gemma:
        return _applyGemmaTemplate(messages);
      case LlamaTemplateType.minicpm:
        return _applyMiniCPMTemplate(messages);
      case LlamaTemplateType.unknown:
        return _applyDefaultTemplate(messages);
    }
  }

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
      if (msg.role == LlamaMessageRole.user) {
        buffer.write('[INST] ${msg.content} [/INST]');
      } else if (msg.role == LlamaMessageRole.assistant) {
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
      if (msg.role == LlamaMessageRole.user) {
        buffer.write('### Instruction:\n${msg.content}\n\n');
      } else if (msg.role == LlamaMessageRole.assistant) {
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

  String _applyGemmaTemplate(List<LlamaChatMessage> messages) {
    final buffer = StringBuffer();

    for (final msg in messages) {
      final role = msg.role.name;
      buffer.write('<start_of_turn>$role\n');
      buffer.write(msg.content.trim());
      buffer.write('<end_of_turn>\n');
    }

    buffer.write('<start_of_turn>model\n');

    return buffer.toString();
  }

  String _applyMiniCPMTemplate(List<LlamaChatMessage> messages) {
    final buffer = StringBuffer();

    for (final msg in messages) {
      if (msg.role == LlamaMessageRole.user) {
        buffer.write('<用户>${msg.content}</用户>\n');
      } else if (msg.role == LlamaMessageRole.assistant) {
        buffer.write('<AI>${msg.content}</AI>\n');
      } else {
        buffer.write('${msg.content}\n');
      }
    }

    buffer.write('<AI>');

    return buffer.toString();
  }
}
