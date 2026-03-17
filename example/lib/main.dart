import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:llama_native/llama_native.dart';
import 'package:path_provider/path_provider.dart';

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
  final List<_DisplayMessage> _messages = [];
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  final LlamaEngine _engine = LlamaEngine();
  LlamaChat? _chat;

  String _statusText = '请选择模型文件开始';

  @override
  void initState() {
    super.initState();
    _engine.onStateChange.listen((state) {
      setState(() {
        switch (state) {
          case LoadState.idle:
            _statusText = '请选择模型文件开始';
            break;
          case LoadState.initializing:
            _statusText = '正在初始化...';
            break;
          case LoadState.loading:
            _statusText = '正在加载模型...';
            break;
          case LoadState.ready:
            _statusText = '模型已就绪';
            _chat = LlamaChat(engine: _engine);
            break;
          case LoadState.error:
            _statusText = '错误: ${_engine.error}';
            break;
        }
      });
    });

    _engine.onProgress.listen((progress) {
      setState(() {
        switch (progress) {
          case LoadProgress.initializing:
            _statusText = '正在初始化...';
            break;
          case LoadProgress.allocatingMemory:
            _statusText = '正在分配内存...';
            break;
          case LoadProgress.loadingModel:
            _statusText = '正在加载模型...';
            break;
          case LoadProgress.creatingContext:
            _statusText = '正在创建上下文...';
            break;
          case LoadProgress.ready:
            _statusText = '模型已就绪';
            break;
        }
      });
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    _scrollController.dispose();
    _chat?.dispose();
    _engine.dispose();
    super.dispose();
  }

  Future<void> _pickModel() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.any, // 强制允许点击所有文件
        allowMultiple: false,
      );

      if (result != null) {
        String path = result.files.single.path!;
        if (path.endsWith('.gguf')) {
          print("成功选中模型: $path");
          // 传给你的 C++ Wrapper
        } else {
          print("请选择正确的 GGUF 文件");
        }
      }

      if (result != null && result.files.single.path != null) {
        String destinationPath;
        if (Platform.isIOS) {
          String originalPath = result.files.single.path!;

          // 1. 获取 App 自己的私有目录
          final docsDir = await getApplicationDocumentsDirectory();
          final fileName = result.files.single.name;
          destinationPath = "${docsDir.path}/$fileName";

          // 2. 如果文件不在私有目录，拷贝一份（或者如果已存在则跳过）
          final sourceFile = File(originalPath);
          final destinationFile = File(destinationPath);

          if (!await destinationFile.exists()) {
            print("正在拷贝模型到沙盒，请稍候...");
            await sourceFile.copy(destinationPath);
          }

          // 3. 将稳定、有权限的沙盒路径传给 C++
          print("加载模型: $destinationPath");
        } else {
          destinationPath = result.files.single.path!;
        }

        await _engine.load(destinationPath);
      }
    } catch (e) {
      setState(() {
        _statusText = '选择模型失败: $e';
      });
    }
  }

  Future<void> _sendMessage() async {
    final text = _controller.text.trim();
    if (text.isEmpty || !_engine.isReady) return;

    _controller.clear();

    setState(() {
      _messages.add(_DisplayMessage(role: LlamaChatMessageRole.user, content: text));
      _messages.add(_DisplayMessage(role: LlamaChatMessageRole.assistant, content: '', isStreaming: true));
    });

    _scrollToBottom();

    try {
      final buffer = StringBuffer();
      await for (final tokenText in _chat!.sendMessage(text)) {
        buffer.write(tokenText);

        setState(() {
          _messages.last = _DisplayMessage(
            role: LlamaChatMessageRole.assistant,
            content: buffer.toString(),
            isStreaming: true,
          );
        });

        _scrollToBottom();
      }

      setState(() {
        _messages.last = _DisplayMessage(
          role: LlamaChatMessageRole.assistant,
          content: buffer.toString(),
          isStreaming: false,
        );
      });
    } catch (e) {
      setState(() {
        _messages.last = _DisplayMessage(
          role: LlamaChatMessageRole.assistant,
          content: '生成回复时出错: $e',
          isStreaming: false,
        );
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
      _chat?.reset();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Llama Chat'),
        actions: [
          IconButton(
            icon: const Icon(Icons.folder_open),
            tooltip: '选择模型',
            onPressed: _engine.state == LoadState.loading ? null : _pickModel,
          ),
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
                          _engine.isReady ? '开始对话吧！' : '请先加载模型',
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
                        hintText: _engine.isReady ? '输入消息...' : '请先加载模型',
                        border: OutlineInputBorder(borderRadius: BorderRadius.circular(24)),
                        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                      ),
                      maxLines: null,
                      textInputAction: TextInputAction.send,
                      onSubmitted: (_) => _sendMessage(),
                      enabled: _engine.isReady,
                    ),
                  ),
                  const SizedBox(width: 8),
                  IconButton.filled(
                    icon: _messages.isNotEmpty && _messages.last.isStreaming
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                          )
                        : const Icon(Icons.send),
                    onPressed: _engine.isReady && (_messages.isEmpty || !_messages.last.isStreaming)
                        ? _sendMessage
                        : null,
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
  final _DisplayMessage message;

  const _MessageBubble({required this.message});

  @override
  Widget build(BuildContext context) {
    final isUser = message.role == LlamaChatMessageRole.user;

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

class _DisplayMessage {
  final LlamaChatMessageRole role;
  final String content;
  final bool isStreaming;

  _DisplayMessage({required this.role, required this.content, this.isStreaming = false});
}
