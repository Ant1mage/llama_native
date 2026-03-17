import 'dart:async';

import 'package:llama_native/llama_engine.dart';

enum MessageRole { user, assistant }

class ChatMessage {
  final MessageRole role;
  final String content;

  const ChatMessage({required this.role, required this.content});

  Map<String, String> toMap() => {'role': role == MessageRole.user ? 'user' : 'assistant', 'content': content};
}

class LlamaChat {
  final LlamaEngine _engine;
  final String _systemPrompt;
  final int _maxTokens;

  final List<ChatMessage> _history = [];
  final _messageController = StreamController<ChatMessage>.broadcast();

  LlamaChat({
    required LlamaEngine engine,
    String systemPrompt = '',
    int maxTokens = 1024,
  })  : _engine = engine,
        _systemPrompt = systemPrompt,
        _maxTokens = maxTokens;

  List<ChatMessage> get history => List.unmodifiable(_history);
  Stream<ChatMessage> get onMessage => _messageController.stream;
  bool get isReady => _engine.isReady;

  void clearHistory() {
    _history.clear();
  }

  Stream<String> sendMessage(String userMessage) async* {
    if (!isReady) {
      throw StateError('Engine not ready');
    }

    _history.add(ChatMessage(role: MessageRole.user, content: userMessage));
    _messageController.add(_history.last);

    final prompt = _buildPrompt();
    final tokens = await _engine.tokenize(prompt, addBos: false);

    final buffer = StringBuffer();
    await for (final text in _engine.generate(tokens, maxTokens: _maxTokens)) {
      buffer.write(text);
      yield text;
    }

    final assistantMessage = buffer.toString();
    _history.add(ChatMessage(role: MessageRole.assistant, content: assistantMessage));
    _messageController.add(_history.last);
  }

  Future<String> sendMessageAndWait(String userMessage) async {
    final buffer = StringBuffer();
    await for (final text in sendMessage(userMessage)) {
      buffer.write(text);
    }
    return buffer.toString();
  }

  String _buildPrompt() {
    final buffer = StringBuffer();

    if (_systemPrompt.isNotEmpty) {
      buffer.write('<|im_start|>system\n$_systemPrompt<|im_end|>\n');
    }

    for (final msg in _history) {
      final role = msg.role == MessageRole.user ? 'user' : 'assistant';
      buffer.write('<|im_start|>$role\n${msg.content}<|im_end|>\n');
    }

    buffer.write('<|im_start|>assistant\n');

    return buffer.toString();
  }

  Future<void> reset() async {
    _history.clear();
    await _engine.reset();
  }

  void dispose() {
    _messageController.close();
  }
}
