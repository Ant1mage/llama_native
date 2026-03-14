import 'dart:io';
import 'package:flutter/material.dart';
import 'package:path/path.dart' as path;
import 'package:llama_native/llama_native.dart';

void main() {
  runApp(const LlamaChatApp());
}

class LlamaChatApp extends StatelessWidget {
  const LlamaChatApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'LLaMA Chat',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF6750A4), brightness: Brightness.light),
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF6750A4), brightness: Brightness.dark),
        useMaterial3: true,
      ),
      home: const ChatPage(),
    );
  }
}

// ── 消息数据模型 ──────────────────────────────────────────
enum MsgRole { user, assistant }

class ChatMsg {
  final MsgRole role;
  String text;
  bool streaming;

  ChatMsg({required this.role, required this.text, this.streaming = false});
}

// ── 模型初始化状态 ────────────────────────────────────────
enum ModelStatus { idle, loading, ready, error }

// ── 主聊天页面 ────────────────────────────────────────────
class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  // llama 资源
  LlamaBackend? _backend;
  LlamaModel? _model;
  LlamaContext? _context;
  ChatTokenizer? _tokenizer;

  // UI 状态
  ModelStatus _status = ModelStatus.idle;
  String _statusMsg = '点击右上角加载模型';
  bool _generating = false;

  // 对话历史（含系统 prompt）
  final List<ChatMessage> _history = [const ChatMessage.system('你是一个有帮助的 AI 助手。')];
  final List<ChatMsg> _messages = [];

  final TextEditingController _inputCtrl = TextEditingController();
  final ScrollController _scrollCtrl = ScrollController();
  final FocusNode _focusNode = FocusNode();

  @override
  void dispose() {
    _inputCtrl.dispose();
    _scrollCtrl.dispose();
    _focusNode.dispose();
    _context?.dispose();
    _model?.dispose();
    _backend?.dispose();
    super.dispose();
  }

  // ── 加载模型 ────────────────────────────────────────────
  Future<void> _loadModel() async {
    setState(() {
      _status = ModelStatus.loading;
      _statusMsg = '正在加载模型…';
    });

    try {
      // 初始化后端
      _backend = LlamaBackend.instance;
      await _backend!.initialize();

      // 定位模型文件（macOS/iOS 场景：项目根目录 models/）
      final exePath = Platform.resolvedExecutable;
      final appDir = Directory(exePath).parent.path;
      // 在 macOS debug 下，往上找几层到工程根
      String modelPath = '';
      final candidates = [
        // macOS debug: .../example/build/macos/...
        path.join(appDir, '../../../../models/Qwen3.5-4B-Q4_K_M.gguf'),
        path.join(appDir, '../../../models/Qwen3.5-4B-Q4_K_M.gguf'),
        path.join(appDir, '../../models/Qwen3.5-4B-Q4_K_M.gguf'),
        path.join(appDir, '../models/Qwen3.5-4B-Q4_K_M.gguf'),
        // 兜底：绝对路径（开发时写死）
        '/Users/alexanderz/Desktop/llama_native/models/Qwen3.5-4B-Q4_K_M.gguf',
      ];
      for (final p in candidates) {
        if (File(p).existsSync()) {
          modelPath = p;
          break;
        }
      }
      if (modelPath.isEmpty) {
        throw Exception('找不到模型文件，请检查 models/ 目录');
      }

      _model = LlamaModel.load(LlamaModelConfig(modelPath: modelPath, vocabOnly: false));

      final cfg = InferenceConfig.defaultMacOS();
      _context = LlamaContext.create(_model!, cfg);

      _tokenizer = ChatTokenizer(model: _model!, templateType: ChatTemplateType.qwen, addBos: true, addEos: false);

      setState(() {
        _status = ModelStatus.ready;
        _statusMsg = '模型已就绪';
      });
    } catch (e) {
      setState(() {
        _status = ModelStatus.error;
        _statusMsg = '加载失败：$e';
      });
    }
  }

  // ── 发送消息并流式接收回复 ───────────────────────────────
  Future<void> _send() async {
    final text = _inputCtrl.text.trim();
    if (text.isEmpty || _generating) return;
    if (_status != ModelStatus.ready) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('请先加载模型')));
      return;
    }

    // 添加用户消息
    _history.add(ChatMessage.user(text));
    setState(() {
      _messages.add(ChatMsg(role: MsgRole.user, text: text));
      _inputCtrl.clear();
      _generating = true;
    });
    _scrollToBottom();

    // 添加助手占位消息（流式填充）
    final assistantMsg = ChatMsg(role: MsgRole.assistant, text: '', streaming: true);
    setState(() => _messages.add(assistantMsg));

    try {
      final inputTokens = _tokenizer!.tokenizeMessages(_history);
      final stream = _context!.generateStream(inputTokens, maxTokens: 512);

      await for (final result in stream) {
        if (result.text.isNotEmpty) {
          setState(() => assistantMsg.text += result.text);
          _scrollToBottom();
        }
        if (result.isEnd) break;
      }

      // 将完整回复加入历史
      _history.add(ChatMessage.assistant(assistantMsg.text));
    } catch (e) {
      setState(() => assistantMsg.text = '❌ 生成失败：$e');
    } finally {
      setState(() {
        assistantMsg.streaming = false;
        _generating = false;
      });
      _scrollToBottom();
      _focusNode.requestFocus();
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollCtrl.hasClients) {
        _scrollCtrl.animateTo(
          _scrollCtrl.position.maxScrollExtent,
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeOut,
        );
      }
    });
  }

  // ── 清空对话 ─────────────────────────────────────────────
  void _clearChat() {
    setState(() {
      _messages.clear();
      _history.removeRange(1, _history.length); // 保留 system prompt
    });
  }

  // ── Build ─────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('🦙 LLaMA Chat'),
        centerTitle: true,
        backgroundColor: scheme.surfaceContainerHighest,
        actions: [
          if (_messages.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.delete_outline),
              tooltip: '清空对话',
              onPressed: _generating ? null : _clearChat,
            ),
          IconButton(
            icon: Icon(
              _status == ModelStatus.ready
                  ? Icons.check_circle
                  : _status == ModelStatus.loading
                  ? Icons.hourglass_top
                  : Icons.download_rounded,
            ),
            tooltip: _status == ModelStatus.ready ? '模型已加载' : '加载模型',
            onPressed: _status == ModelStatus.loading || _status == ModelStatus.ready ? null : _loadModel,
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: Column(
        children: [
          // 状态栏
          _StatusBar(status: _status, message: _statusMsg),

          // 消息列表
          Expanded(
            child: _messages.isEmpty
                ? _EmptyHint(onLoad: _status == ModelStatus.idle ? _loadModel : null)
                : ListView.builder(
                    controller: _scrollCtrl,
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    itemCount: _messages.length,
                    itemBuilder: (ctx, i) => _MessageBubble(msg: _messages[i], scheme: scheme),
                  ),
          ),

          // 输入栏
          _InputBar(
            controller: _inputCtrl,
            focusNode: _focusNode,
            enabled: _status == ModelStatus.ready && !_generating,
            generating: _generating,
            onSend: _send,
          ),
        ],
      ),
    );
  }
}

// ── 状态栏 ────────────────────────────────────────────────
class _StatusBar extends StatelessWidget {
  final ModelStatus status;
  final String message;

  const _StatusBar({required this.status, required this.message});

  @override
  Widget build(BuildContext context) {
    Color bg;
    switch (status) {
      case ModelStatus.ready:
        bg = Colors.green.shade700;
        break;
      case ModelStatus.loading:
        bg = Colors.orange.shade700;
        break;
      case ModelStatus.error:
        bg = Colors.red.shade700;
        break;
      default:
        bg = Colors.grey.shade600;
    }

    return Container(
      width: double.infinity,
      color: bg,
      padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 12),
      child: Row(
        children: [
          if (status == ModelStatus.loading)
            const SizedBox(
              width: 12,
              height: 12,
              child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
            ),
          if (status == ModelStatus.loading) const SizedBox(width: 8),
          Expanded(
            child: Text(message, style: const TextStyle(color: Colors.white, fontSize: 12)),
          ),
        ],
      ),
    );
  }
}

// ── 空状态提示 ────────────────────────────────────────────
class _EmptyHint extends StatelessWidget {
  final VoidCallback? onLoad;
  const _EmptyHint({this.onLoad});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text('🦙', style: TextStyle(fontSize: 64)),
          const SizedBox(height: 12),
          const Text('LLaMA Native Chat', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          const Text('本地大模型，流式输出，隐私优先', style: TextStyle(color: Colors.grey)),
          if (onLoad != null) ...[
            const SizedBox(height: 24),
            FilledButton.icon(onPressed: onLoad, icon: const Icon(Icons.download_rounded), label: const Text('加载模型')),
          ],
        ],
      ),
    );
  }
}

// ── 消息气泡 ──────────────────────────────────────────────
class _MessageBubble extends StatelessWidget {
  final ChatMsg msg;
  final ColorScheme scheme;

  const _MessageBubble({required this.msg, required this.scheme});

  @override
  Widget build(BuildContext context) {
    final isUser = msg.role == MsgRole.user;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        mainAxisAlignment: isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        children: [
          if (!isUser) ...[
            CircleAvatar(
              radius: 16,
              backgroundColor: scheme.primaryContainer,
              child: const Text('🦙', style: TextStyle(fontSize: 14)),
            ),
            const SizedBox(width: 8),
          ],
          Flexible(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: isUser ? scheme.primary : scheme.surfaceContainerHigh,
                borderRadius: BorderRadius.only(
                  topLeft: const Radius.circular(18),
                  topRight: const Radius.circular(18),
                  bottomLeft: Radius.circular(isUser ? 18 : 4),
                  bottomRight: Radius.circular(isUser ? 4 : 18),
                ),
                boxShadow: [
                  BoxShadow(color: Colors.black.withOpacity(0.06), blurRadius: 4, offset: const Offset(0, 2)),
                ],
              ),
              child: msg.streaming && msg.text.isEmpty
                  ? _TypingIndicator(color: scheme.onSurface)
                  : Row(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Flexible(
                          child: SelectableText(
                            msg.text,
                            style: TextStyle(
                              color: isUser ? scheme.onPrimary : scheme.onSurface,
                              fontSize: 15,
                              height: 1.4,
                            ),
                          ),
                        ),
                        if (msg.streaming) ...[
                          const SizedBox(width: 6),
                          _CursorBlink(color: isUser ? scheme.onPrimary : scheme.primary),
                        ],
                      ],
                    ),
            ),
          ),
          if (isUser) ...[
            const SizedBox(width: 8),
            CircleAvatar(
              radius: 16,
              backgroundColor: scheme.secondaryContainer,
              child: Icon(Icons.person, size: 18, color: scheme.onSecondaryContainer),
            ),
          ],
        ],
      ),
    );
  }
}

// ── 打字中动画（三点） ──────────────────────────────────────
class _TypingIndicator extends StatefulWidget {
  final Color color;
  const _TypingIndicator({required this.color});

  @override
  State<_TypingIndicator> createState() => _TypingIndicatorState();
}

class _TypingIndicatorState extends State<_TypingIndicator> with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 900))..repeat();
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _ctrl,
      builder: (_, __) {
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: List.generate(3, (i) {
            final delay = i / 3;
            final t = (_ctrl.value - delay).clamp(0.0, 1.0);
            final opacity = (t < 0.5 ? t * 2 : (1 - t) * 2).clamp(0.3, 1.0);
            return Container(
              margin: const EdgeInsets.symmetric(horizontal: 2),
              width: 6,
              height: 6,
              decoration: BoxDecoration(color: widget.color.withOpacity(opacity), shape: BoxShape.circle),
            );
          }),
        );
      },
    );
  }
}

// ── 流式输出光标闪烁 ──────────────────────────────────────
class _CursorBlink extends StatefulWidget {
  final Color color;
  const _CursorBlink({required this.color});

  @override
  State<_CursorBlink> createState() => _CursorBlinkState();
}

class _CursorBlinkState extends State<_CursorBlink> with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 600))..repeat(reverse: true);
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _ctrl,
      child: Container(width: 2, height: 16, color: widget.color),
    );
  }
}

// ── 输入栏 ────────────────────────────────────────────────
class _InputBar extends StatelessWidget {
  final TextEditingController controller;
  final FocusNode focusNode;
  final bool enabled;
  final bool generating;
  final VoidCallback onSend;

  const _InputBar({
    required this.controller,
    required this.focusNode,
    required this.enabled,
    required this.generating,
    required this.onSend,
  });

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;

    return SafeArea(
      child: Container(
        padding: const EdgeInsets.fromLTRB(12, 8, 8, 12),
        decoration: BoxDecoration(
          color: scheme.surface,
          border: Border(top: BorderSide(color: scheme.outlineVariant, width: 0.5)),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Expanded(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxHeight: 140),
                child: TextField(
                  controller: controller,
                  focusNode: focusNode,
                  enabled: enabled,
                  maxLines: null,
                  keyboardType: TextInputType.multiline,
                  textInputAction: TextInputAction.newline,
                  style: const TextStyle(fontSize: 15),
                  decoration: InputDecoration(
                    hintText: enabled
                        ? '输入消息…'
                        : generating
                        ? '正在生成…'
                        : '请先加载模型',
                    contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                    filled: true,
                    fillColor: scheme.surfaceContainerHigh,
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(24), borderSide: BorderSide.none),
                  ),
                  onSubmitted: enabled ? (_) => onSend() : null,
                ),
              ),
            ),
            const SizedBox(width: 8),
            AnimatedSwitcher(
              duration: const Duration(milliseconds: 200),
              child: generating
                  ? Container(
                      key: const ValueKey('loading'),
                      width: 44,
                      height: 44,
                      padding: const EdgeInsets.all(10),
                      child: CircularProgressIndicator(strokeWidth: 2.5, color: scheme.primary),
                    )
                  : FilledButton(
                      key: const ValueKey('send'),
                      onPressed: enabled ? onSend : null,
                      style: FilledButton.styleFrom(
                        minimumSize: const Size(44, 44),
                        padding: EdgeInsets.zero,
                        shape: const CircleBorder(),
                      ),
                      child: const Icon(Icons.send_rounded, size: 20),
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
