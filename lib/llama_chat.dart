import 'dart:async';

import 'package:llama_native/llama_engine.dart';
import 'package:llama_native/llama_chat_message.dart';

export 'package:llama_native/llama_chat_message.dart' show LlamaChatMessage, LlamaMessageRole;

class LlamaChat {
  final LlamaEngine _engine;
  final String _systemPrompt;
  final int _maxTokens;

  final List<LlamaChatMessage> _history = [];
  final _messageController = StreamController<LlamaChatMessage>.broadcast();

  LlamaChat({required LlamaEngine engine, String systemPrompt = '', int maxTokens = 1024})
    : _engine = engine,
      _systemPrompt = systemPrompt,
      _maxTokens = maxTokens;

  List<LlamaChatMessage> get history => List.unmodifiable(_history);
  Stream<LlamaChatMessage> get onMessage => _messageController.stream;
  bool get isReady => _engine.isReady;

  void clearHistory() {
    _history.clear();
  }

  Stream<String> sendMessage(String userMessage) async* {
    if (!isReady) {
      throw StateError('Engine not ready');
    }

    _history.add(LlamaChatMessage.user(userMessage));
    _messageController.add(_history.last);

    final prompt = await _buildPrompt();
    final tokens = await _engine.tokenize(prompt, addBos: false);

    final buffer = StringBuffer();
    await for (final generation in _engine.generate(tokens, maxTokens: _maxTokens)) {
      buffer.write(generation.text);
      yield generation.text;
      if (generation.isEnd) break;
    }

    final assistantMessage = buffer.toString();
    _history.add(LlamaChatMessage.assistant(assistantMessage));
    _messageController.add(_history.last);
  }

  Future<String> sendMessageAndWait(String userMessage) async {
    final buffer = StringBuffer();
    await for (final text in sendMessage(userMessage)) {
      buffer.write(text);
    }
    return buffer.toString();
  }

  Future<String> _buildPrompt() async {
    final messages = <Map<String, String>>[];

    if (_systemPrompt.isNotEmpty) {
      messages.add({'role': 'system', 'content': _systemPrompt});
    }

    for (final msg in _history) {
      messages.add(msg.toMap());
    }

    return _engine.applyChatTemplate(messages);
  }

  Future<void> reset() async {
    _history.clear();
    await _engine.reset();
  }

  void dispose() {
    _messageController.close();
  }
}
