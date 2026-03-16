import 'dart:async';

import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:llama_native/llama_native.dart';

void main() {
  runApp(const LlamaChatApp());
}

class LlamaChatApp extends StatelessWidget {
  const LlamaChatApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Llama Chat',
      theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue), useMaterial3: true),
      home: const ChatPage(),
    );
  }
}

class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final List<AppChatMessage> _messages = [];
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  LlamaModel? _model;
  LlamaContext? _context;
  bool _isLoading = false;
  bool _isModelLoaded = false;
  String? _modelPath;
  String _statusText = '请选择模型文件开始';

  @override
  void dispose() {
    _controller.dispose();
    _scrollController.dispose();
    _context?.dispose();
    _model?.dispose();
    LlamaBackend.instance.dispose();
    super.dispose();
  }

  Future<void> _pickModel() async {
    try {
      final result = await FilePicker.platform.pickFiles(type: FileType.custom, allowedExtensions: ['gguf']);

      if (result != null && result.files.single.path != null) {
        setState(() {
          _modelPath = result.files.single.path;
          _statusText = '已选择模型: ${result.files.single.name}';
        });
        await _loadModel();
      }
    } catch (e) {
      setState(() {
        _statusText = '选择模型失败: $e';
      });
    }
  }

  Future<void> _loadModel() async {
    if (_modelPath == null) return;

    setState(() {
      _isLoading = true;
      _statusText = '正在加载模型...';
    });

    try {
      final backend = LlamaBackend.instance;
      await backend.initialize();

      final modelConfig = LlamaModelConfig(modelPath: _modelPath!);
      _model = LlamaModel.load(modelConfig);

      final inferenceConfig = InferenceConfig.defaults(
        nCtx: 4096,
        sampling: SamplingConfig(temperature: 0.7, topP: 0.9, topK: 40),
      );

      _context = LlamaContext.create(_model!, inferenceConfig);

      setState(() {
        _isModelLoaded = true;
        _isLoading = false;
        _statusText = '模型加载成功，可以开始对话';
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _statusText = '加载模型失败: $e';
      });
    }
  }

  Future<void> _sendMessage() async {
    final text = _controller.text.trim();
    if (text.isEmpty || !_isModelLoaded || _isLoading) return;

    _controller.clear();

    setState(() {
      _messages.add(AppChatMessage(role: MessageRole.user, content: text));
      _messages.add(AppChatMessage(role: MessageRole.assistant, content: '', isStreaming: true));
      _isLoading = true;
    });

    _scrollToBottom();

    try {
      final chatTokenizer = ChatTokenizer(model: _model!, templateType: ChatTemplateType.chatml);

      final chatMessages = _messages
          .where((m) => !m.isStreaming)
          .map((m) => ChatMessage(role: m.role, content: m.content))
          .toList();

      final prompt = chatTokenizer.applyTemplate(chatMessages);
      debugPrint('=== Prompt ===\n$prompt\n=============');

      final tokens = _model!.tokenize(prompt, addBos: false);
      debugPrint('=== Tokens (${tokens.length}) ===\n$tokens\n=============');

      final stream = _context!.generateStream(tokens, maxTokens: 512);

      final buffer = StringBuffer();
      await for (final generation in stream) {
        buffer.write(generation.text);

        setState(() {
          _messages.last = AppChatMessage(
            role: MessageRole.assistant,
            content: buffer.toString(),
            isStreaming: !generation.isEnd,
          );
        });

        _scrollToBottom();

        if (generation.isEnd) break;
      }

      setState(() {
        _messages.last = AppChatMessage(role: MessageRole.assistant, content: buffer.toString(), isStreaming: false);
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _messages.last = AppChatMessage(role: MessageRole.assistant, content: '生成回复时出错: $e', isStreaming: false);
        _isLoading = false;
      });
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _clearChat() {
    setState(() {
      _messages.clear();
      _context?.reset();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Llama Chat'),
        actions: [
          IconButton(icon: const Icon(Icons.folder_open), tooltip: '选择模型', onPressed: _isLoading ? null : _pickModel),
          IconButton(
            icon: const Icon(Icons.delete_outline),
            tooltip: '清空对话',
            onPressed: _messages.isEmpty ? null : _clearChat,
          ),
        ],
      ),
      body: Column(
        children: [
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(8),
            color: Theme.of(context).colorScheme.surfaceContainerHighest,
            child: Text(_statusText, style: Theme.of(context).textTheme.bodySmall, textAlign: TextAlign.center),
          ),
          Expanded(
            child: _messages.isEmpty
                ? Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.chat_bubble_outline, size: 64, color: Theme.of(context).colorScheme.outline),
                        const SizedBox(height: 16),
                        Text(
                          _isModelLoaded ? '开始对话吧！' : '请先加载模型',
                          style: Theme.of(
                            context,
                          ).textTheme.bodyLarge?.copyWith(color: Theme.of(context).colorScheme.outline),
                        ),
                      ],
                    ),
                  )
                : ListView.builder(
                    controller: _scrollController,
                    padding: const EdgeInsets.all(16),
                    itemCount: _messages.length,
                    itemBuilder: (context, index) {
                      final message = _messages[index];
                      return _MessageBubble(message: message);
                    },
                  ),
          ),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surface,
              boxShadow: [
                BoxShadow(color: Colors.black.withValues(alpha: 0.05), blurRadius: 10, offset: const Offset(0, -5)),
              ],
            ),
            child: SafeArea(
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _controller,
                      decoration: InputDecoration(
                        hintText: _isModelLoaded ? '输入消息...' : '请先加载模型',
                        border: OutlineInputBorder(borderRadius: BorderRadius.circular(24)),
                        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                      ),
                      maxLines: null,
                      textInputAction: TextInputAction.send,
                      onSubmitted: (_) => _sendMessage(),
                      enabled: _isModelLoaded && !_isLoading,
                    ),
                  ),
                  const SizedBox(width: 8),
                  IconButton.filled(
                    icon: _isLoading
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                          )
                        : const Icon(Icons.send),
                    onPressed: _isLoading || !_isModelLoaded ? null : _sendMessage,
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _MessageBubble extends StatelessWidget {
  final AppChatMessage message;

  const _MessageBubble({required this.message});

  @override
  Widget build(BuildContext context) {
    final isUser = message.role == MessageRole.user;

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.75),
        decoration: BoxDecoration(
          color: isUser ? Theme.of(context).colorScheme.primary : Theme.of(context).colorScheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              message.content,
              style: TextStyle(
                color: isUser ? Theme.of(context).colorScheme.onPrimary : Theme.of(context).colorScheme.onSurface,
              ),
            ),
            if (message.isStreaming)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: SizedBox(
                  width: 12,
                  height: 12,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    color: isUser ? Theme.of(context).colorScheme.onPrimary : Theme.of(context).colorScheme.primary,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class AppChatMessage {
  final MessageRole role;
  final String content;
  final bool isStreaming;

  AppChatMessage({required this.role, required this.content, this.isStreaming = false});
}
