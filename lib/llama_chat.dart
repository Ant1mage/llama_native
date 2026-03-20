import 'dart:async';

import 'package:llama_native/llama_engine.dart';
import 'package:llama_native/llama_chat_message.dart';
import 'package:llama_native/src/engine/exceptions/llama_exceptions.dart';
import 'package:llama_native/src/log/logger.dart';

export 'package:llama_native/llama_chat_message.dart' show LlamaChatMessage, LlamaMessageRole;

class LlamaChat {
  final LlamaEngine _engine;
  final String _systemPrompt;
  final int _maxTokens;
  final Logger _logger = Logger('LlamaChat');

  final List<LlamaChatMessage> _history = [];
  final _messageController = StreamController<LlamaChatMessage>.broadcast();

  bool _systemPromptProcessed = false;
  List<int> _systemPromptTokens = [];

  LlamaChat({required LlamaEngine engine, String systemPrompt = '', int maxTokens = 1024})
    : _engine = engine,
      _systemPrompt = systemPrompt,
      _maxTokens = maxTokens;

  List<LlamaChatMessage> get history => List.unmodifiable(_history);
  Stream<LlamaChatMessage> get onMessage => _messageController.stream;
  bool get isReady => _engine.isReady;
  bool get isGenerating => _engine.isGenerating;
  int get systemPromptTokensCount => _systemPromptTokens.length;
  bool get systemPromptProcessed => _systemPromptProcessed;

  void clearHistory() {
    _logger.debug('清空历史记录');
    _history.clear();
    _systemPromptProcessed = false;
  }

  void stop() {
    _logger.info('停止生成');
    _engine.stop();
  }

  Stream<String> sendMessage(String userMessage) async* {
    if (!isReady) {
      throw StateError('Engine not ready');
    }

    _logger.info('发送消息: ${userMessage.length}字符');
    _history.add(LlamaChatMessage.user(userMessage));
    _messageController.add(_history.last);

    if (!_systemPromptProcessed && _systemPrompt.isNotEmpty) {
      _logger.debug('处理系统提示词');
      final systemOnlyPrompt = await _engine.applyChatTemplate([
        {'role': 'system', 'content': _systemPrompt},
      ]);
      _systemPromptTokens = await _engine.tokenize(systemOnlyPrompt, addBos: false);

      if (_systemPromptTokens.isNotEmpty) {
        await _engine.setKeepPrefixTokens(_systemPromptTokens);
        _logger.debug('系统提示词Token数: ${_systemPromptTokens.length}');
      }

      _systemPromptProcessed = true;
    }

    final prompt = await _buildPrompt();
    final tokens = await _engine.tokenize(prompt, addBos: false);
    _logger.debug('提示词Token数: ${tokens.length}');

    final buffer = StringBuffer();
    try {
      await for (final generation in _engine.generate(tokens, maxTokens: _maxTokens)) {
        buffer.write(generation.text);
        yield generation.text;
        if (generation.isEnd) break;
      }
    } on LlamaException catch (e) {
      if (e.type == LlamaErrorType.kvCache) {
        _logger.warning('KV缓存已满，重置引擎');
        await _engine.reset();
      }
      rethrow;
    }

    final assistantMessage = buffer.toString();
    _history.add(LlamaChatMessage.assistant(assistantMessage));
    _messageController.add(_history.last);
    _logger.info('生成完成: ${assistantMessage.length}字符');
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
    _logger.info('重置聊天');
    _history.clear();
    _systemPromptProcessed = false;
    await _engine.reset();
  }

  void dispose() {
    _logger.info('释放聊天资源');
    _messageController.close();
  }
}
