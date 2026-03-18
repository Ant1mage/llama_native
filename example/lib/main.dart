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

  List<String> _statusLines = ['请选择模型文件开始'];
  String? _modelPath;
  String _selectedModel = '';
  bool _isSummarizing = false;

  PerformanceMetrics _performanceMetrics = PerformanceMetrics.empty();
  List<double>? _lastEmbedding;
  String _embeddingInput = '';

  List<String> _getDeviceInfo() {
    final info = PlatformInfo.getHardwareInfo().split('\n');
    return ['Model: $_selectedModel', ...info].where((line) => line.trim().isNotEmpty).toList();
  }

  @override
  void initState() {
    super.initState();
    _engine.onStateChange.listen((state) {
      setState(() {
        switch (state) {
          case LoadState.idle:
            _statusLines = ['请选择模型文件开始'];
            break;
          case LoadState.initializing:
            _statusLines = ['正在初始化...'];
            break;
          case LoadState.loading:
            _statusLines = ['正在加载模型...'];
            break;
          case LoadState.ready:
            _statusLines = _getDeviceInfo();
            _chat = LlamaChat(
              engine: _engine,
              systemPrompt: "你叫Lumen, 是一个专业的智能助手, 每次回答不得超于4096字, **去掉思考过程**, 请严格按照这个指示",
            );
            break;
          case LoadState.error:
            _statusLines = ['错误: ${_engine.error}'];
            break;
        }
      });
    });

    _engine.onProgress.listen((progress) {
      setState(() {
        switch (progress) {
          case LoadProgress.initializing:
            _statusLines = ['正在初始化...'];
            break;
          case LoadProgress.allocatingMemory:
            _statusLines = ['正在分配内存...'];
            break;
          case LoadProgress.loadingModel:
            _statusLines = ['正在加载模型...'];
            break;
          case LoadProgress.creatingContext:
            _statusLines = ['正在创建上下文...'];
            break;
          case LoadProgress.ready:
            _statusLines = _getDeviceInfo();
            break;
        }
      });
    });
  }

  Future<String> _generateSummary(String conversationText) async {
    if (_modelPath == null) return '';

    setState(() {
      _isSummarizing = true;
      _statusLines = ['正在生成对话摘要...'];
    });

    try {
      final summaryEngine = LlamaEngine();
      await summaryEngine.load(_modelPath!);

      if (!summaryEngine.isReady) {
        summaryEngine.dispose();
        return '';
      }

      final summaryPrompt =
          '''请用简洁的语言总结以下对话内容，保留关键信息，不超过200字。只输出摘要内容，不要输出其他内容。

对话内容：
$conversationText

摘要：''';

      final tokens = await summaryEngine.tokenize(summaryPrompt, addBos: false);

      final buffer = StringBuffer();
      await for (final generation in summaryEngine.generate(tokens, maxTokens: 256)) {
        buffer.write(generation.text);
        if (generation.isEnd) break;
      }

      await summaryEngine.dispose();

      final summary = buffer.toString().trim();
      debugPrint('生成摘要: $summary');

      setState(() {
        _statusLines = _getDeviceInfo();
        _isSummarizing = false;
      });

      return summary;
    } catch (e) {
      debugPrint('生成摘要失败: $e');
      setState(() {
        _statusLines = _getDeviceInfo();
        _isSummarizing = false;
      });
      return '';
    }
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
      FilePickerResult? result = await FilePicker.platform.pickFiles(type: FileType.any, allowMultiple: false);

      if (result != null) {
        String path = result.files.single.path!;
        if (path.endsWith('.gguf')) {
          debugPrint("成功选中模型: $path");
        } else {
          debugPrint("请选择正确的 GGUF 文件");
        }
      }

      if (result != null && result.files.single.path != null) {
        _selectedModel = result.files.single.name;
        String destinationPath;
        if (Platform.isIOS) {
          String originalPath = result.files.single.path!;

          final docsDir = await getApplicationDocumentsDirectory();
          final fileName = result.files.single.name;
          destinationPath = "${docsDir.path}/$fileName";

          final sourceFile = File(originalPath);
          final destinationFile = File(destinationPath);

          if (!await destinationFile.exists()) {
            debugPrint("正在拷贝模型到沙盒，请稍候...");
            await sourceFile.copy(destinationPath);
          }

          debugPrint("加载模型: $destinationPath");
        } else {
          destinationPath = result.files.single.path!;
        }

        _modelPath = destinationPath;
        await _engine.load(destinationPath);
      }
    } catch (e) {
      setState(() {
        _statusLines = ['选择模型失败: $e'];
      });
    }
  }

  Future<void> _sendMessage() async {
    final text = _controller.text.trim();
    if (text.isEmpty || !_engine.isReady || _isSummarizing) return;

    _controller.clear();

    setState(() {
      _messages.add(_DisplayMessage(role: LlamaMessageRole.user, content: text));
      _messages.add(_DisplayMessage(role: LlamaMessageRole.assistant, content: '', isStreaming: true));
    });

    _scrollToBottom();

    try {
      final buffer = StringBuffer();
      await for (final tokenText in _chat!.sendMessage(text)) {
        buffer.write(tokenText);

        setState(() {
          _messages.last = _DisplayMessage(
            role: LlamaMessageRole.assistant,
            content: buffer.toString(),
            isStreaming: true,
          );
        });

        _scrollToBottom();
      }

      setState(() {
        _messages.last = _DisplayMessage(
          role: LlamaMessageRole.assistant,
          content: buffer.toString(),
          isStreaming: false,
        );
      });

      final metrics = await _engine.getPerformanceMetrics();
      setState(() {
        _performanceMetrics = metrics;
      });
    } catch (e) {
      setState(() {
        _messages.last = _DisplayMessage(role: LlamaMessageRole.assistant, content: '生成回复时出错: $e', isStreaming: false);
      });
    }
  }

  Future<void> _computeEmbedding() async {
    if (!_engine.isReady || _embeddingInput.isEmpty) return;

    try {
      final embedding = await _engine.embed(_embeddingInput);
      setState(() {
        _lastEmbedding = embedding;
      });
    } catch (e) {
      debugPrint('Embedding error: $e');
    }
  }

  void _stopGeneration() {
    _chat?.stop();
    setState(() {
      if (_messages.isNotEmpty && _messages.last.isStreaming) {
        _messages.last = _DisplayMessage(
          role: _messages.last.role,
          content: _messages.last.content,
          isStreaming: false,
        );
      }
    });
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
      _performanceMetrics = PerformanceMetrics.empty();
      _lastEmbedding = null;
    });
  }

  bool get _isGenerating => _messages.isNotEmpty && _messages.last.isStreaming;

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
          _buildStatusPanel(),
          _buildPerformancePanel(),
          _buildEmbeddingPanel(),
          Expanded(
            child: _messages.isEmpty
                ? Center(
                    child: SingleChildScrollView(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
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
                      enabled: _engine.isReady && !_isGenerating && !_isSummarizing,
                    ),
                  ),
                  const SizedBox(width: 8),
                  if (_isGenerating || _isSummarizing)
                    IconButton.filled(
                      icon: _isSummarizing
                          ? const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                            )
                          : const Icon(Icons.stop),
                      style: IconButton.styleFrom(
                        backgroundColor: Theme.of(context).colorScheme.error,
                        foregroundColor: Theme.of(context).colorScheme.onError,
                      ),
                      onPressed: _isSummarizing ? null : _stopGeneration,
                    )
                  else
                    IconButton.filled(icon: const Icon(Icons.send), onPressed: _engine.isReady ? _sendMessage : null),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusPanel() {
    final colors = [Colors.blueAccent, Colors.purpleAccent, Colors.blueGrey];

    return Wrap(
      spacing: 8,
      runSpacing: 4.0,
      children: _statusLines.asMap().entries.map((entry) {
        final index = entry.key;
        final line = entry.value;
        final bgColor = colors[index % colors.length];
        final fgColor = Colors.white;

        return Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.all(Radius.circular(6)),
            color: bgColor,
          ),
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          // color: bgColor,
          child: Text(
            line,
            style: TextStyle(fontSize: 13, color: fgColor),
            textAlign: TextAlign.center,
          ),
        );
      }).toList(),
    );
  }

  Widget _buildPerformancePanel() {
    if (!_engine.isReady) return const SizedBox.shrink();

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      color: _performanceMetrics.isOverloaded
          ? Colors.red.withValues(alpha: 0.1)
          : Theme.of(context).colorScheme.surfaceContainerLow,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _buildMetricItem('ms/tok', _performanceMetrics.msPerToken.toStringAsFixed(1)),
          _buildMetricItem('t/s', _performanceMetrics.tokensPerSecond.toStringAsFixed(1)),
          _buildMetricItem('Prompt', _performanceMetrics.nPromptEval.toString()),
          _buildMetricItem('Eval', _performanceMetrics.nEval.toString()),
          if (_performanceMetrics.isOverloaded)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.red.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(4),
              ),
              child: const Text('过载', style: TextStyle(color: Colors.red, fontSize: 12)),
            ),
        ],
      ),
    );
  }

  Widget _buildMetricItem(String label, String value) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(value, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
        Text(label, style: TextStyle(fontSize: 10, color: Theme.of(context).colorScheme.outline)),
      ],
    );
  }

  Widget _buildEmbeddingPanel() {
    if (!_engine.isReady) return const SizedBox.shrink();

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      color: Theme.of(context).colorScheme.surfaceContainerLow,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.scatter_plot, size: 16),
              const SizedBox(width: 8),
              Text('Embedding', style: Theme.of(context).textTheme.titleSmall),
              const Spacer(),
              if (_lastEmbedding != null)
                Text(
                  'dim: ${_lastEmbedding!.length}',
                  style: TextStyle(fontSize: 12, color: Theme.of(context).colorScheme.outline),
                ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: TextField(
                  decoration: InputDecoration(
                    hintText: '输入文本计算 Embedding',
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                    contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    isDense: true,
                  ),
                  style: const TextStyle(fontSize: 14),
                  onChanged: (value) => _embeddingInput = value,
                  onSubmitted: (_) => _computeEmbedding(),
                ),
              ),
              const SizedBox(width: 8),
              IconButton(
                icon: const Icon(Icons.play_arrow),
                onPressed: _embeddingInput.isNotEmpty ? _computeEmbedding : null,
                style: IconButton.styleFrom(backgroundColor: Theme.of(context).colorScheme.primaryContainer),
              ),
            ],
          ),
          if (_lastEmbedding != null) ...[const SizedBox(height: 8), _buildEmbeddingPreview()],
        ],
      ),
    );
  }

  Widget _buildEmbeddingPreview() {
    final preview = _lastEmbedding!.take(8).map((v) => v.toStringAsFixed(3)).join(', ');
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(color: Theme.of(context).colorScheme.surface, borderRadius: BorderRadius.circular(4)),
      child: Text(
        '[$preview, ...]',
        style: TextStyle(fontSize: 11, fontFamily: 'monospace', color: Theme.of(context).colorScheme.outline),
      ),
    );
  }
}

class _MessageBubble extends StatelessWidget {
  final _DisplayMessage message;

  const _MessageBubble({required this.message});

  @override
  Widget build(BuildContext context) {
    final isUser = message.role == LlamaMessageRole.user;

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
  final LlamaMessageRole role;
  final String content;
  final bool isStreaming;

  _DisplayMessage({required this.role, required this.content, this.isStreaming = false});
}
