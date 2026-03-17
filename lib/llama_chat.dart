import 'dart:async';

import 'package:llama_native/llama_engine.dart';
import 'package:llama_native/llama_chat_message.dart';

export 'package:llama_native/llama_chat_message.dart' show LlamaChatMessage, LlamaMessageRole;

class LlamaChat {
  final LlamaEngine _engine;
  final String _systemPrompt;
  final int _maxTokens;
  final Future<String> Function(String conversationText)? _summarizeCallback;

  final List<LlamaChatMessage> _history = [];
  final _messageController = StreamController<LlamaChatMessage>.broadcast();

  bool _systemPromptProcessed = false;
  List<int> _systemPromptTokens = [];
  String _conversationSummary = '';

  LlamaChat({
    required LlamaEngine engine,
    String systemPrompt = '',
    int maxTokens = 1024,
    Future<String> Function(String conversationText)? summarizeCallback,
  }) : _engine = engine,
       _systemPrompt = systemPrompt,
       _maxTokens = maxTokens,
       _summarizeCallback = summarizeCallback;

  List<LlamaChatMessage> get history => List.unmodifiable(_history);
  Stream<LlamaChatMessage> get onMessage => _messageController.stream;
  bool get isReady => _engine.isReady;
  bool get isGenerating => _engine.isGenerating;
  int get systemPromptTokensCount => _systemPromptTokens.length;
  bool get systemPromptProcessed => _systemPromptProcessed;
  String get conversationSummary => _conversationSummary;

  void clearHistory() {
    _history.clear();
    _systemPromptProcessed = false;
    _conversationSummary = '';
  }

  void stop() {
    _engine.stop();
  }

  Stream<String> sendMessage(String userMessage) async* {
    if (!isReady) {
      throw StateError('Engine not ready');
    }

    _history.add(LlamaChatMessage.user(userMessage));
    _messageController.add(_history.last);

    if (!_systemPromptProcessed && _systemPrompt.isNotEmpty) {
      final systemOnlyPrompt = await _engine.applyChatTemplate([
        {'role': 'system', 'content': _systemPrompt},
      ]);
      _systemPromptTokens = await _engine.tokenize(systemOnlyPrompt, addBos: false);

      if (_systemPromptTokens.isNotEmpty) {
        await _engine.setKeepPrefixTokens(_systemPromptTokens);
      }

      _systemPromptProcessed = true;
    }

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

    final checkResult = await _engine.checkNeedsSummarization();
    if (checkResult['needsSummarization'] == true) {
      final conversationText = checkResult['text'] as String?;
      if (conversationText != null && conversationText.isNotEmpty && _summarizeCallback != null) {
        final summary = await _summarizeCallback(conversationText);
        if (summary.isNotEmpty) {
          _conversationSummary = summary;
          await _engine.applySummary(summary);
        }
      }
    }
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
    _systemPromptProcessed = false;
    _conversationSummary = '';
    await _engine.reset();
  }

  void dispose() {
    _messageController.close();
  }
}
