# Llama Native

Flutter FFI 插件，提供 llama.cpp 的全平台 Dart 封装，支持本地大模型推理。

## 特性

- 🚀 **全平台支持**: Android, iOS, macOS, Linux, Windows
- 📦 **Native Assets**: 自动下载预编译库，无需本地编译
- 💬 **简洁 API**: LlamaEngine + LlamaChat 两层封装
- ⚡ **流式生成**: 实时 token 输出，不阻塞 UI
- 🎯 **TokenGeneration**: 结构化生成结果，包含 token ID、文本、结束状态
- 🔧 **统一异常**: LlamaException + LlamaErrorType 枚举
- 🧠 **智能上下文压缩**: Context 溢出时自动生成摘要，保留关键对话信息
- 🔄 **KV Cache 重建**: 支持摘要 + System Prompt + 最近对话的智能重建

## 快速开始

### 1. 添加依赖

```yaml
dependencies:
  llama_native: ^0.0.1
```

### 2. 基础使用

```dart
import 'package:llama_native/llama_native.dart';

// 创建引擎
final engine = LlamaEngine();

// 加载模型
final success = await engine.load('path/to/model.gguf');
if (!success) {
  print('加载失败: ${engine.error}');
  return;
}

// 监听加载进度
engine.onProgress.listen((progress) {
  print('进度: $progress');
});

// 生成文本
final tokens = await engine.tokenize('你好', addBos: true);
await for (final gen in engine.generate(tokens, maxTokens: 256)) {
  print('Token ${gen.token}: ${gen.text}');
  if (gen.isEnd) break;
}

// 清理
await engine.dispose();
```

### 3. 对话示例

```dart
import 'package:llama_native/llama_native.dart';

// 创建引擎
final engine = LlamaEngine();
await engine.load('path/to/model.gguf');

// 创建对话
final chat = LlamaChat(
  engine: engine,
  systemPrompt: '你是一个 AI 助手',
  maxTokens: 1024,
);

// 发送消息（流式）
await for (final text in chat.sendMessage('你好')) {
  print(text);
}

// 停止生成
chat.stop();

// 检查是否正在生成
if (chat.isGenerating) {
  chat.stop();
}

// 发送消息（等待完成）
final reply = await chat.sendMessageAndWait('介绍一下自己');
print(reply);

// 查看历史
for (final msg in chat.history) {
  print('${msg.role}: ${msg.content}');
}

// 清理
chat.dispose();
await engine.dispose();
```

### 4. 监听状态

```dart
// 监听引擎状态
engine.onStateChange.listen((state) {
  switch (state) {
    case LoadState.idle:
      print('空闲');
    case LoadState.initializing:
      print('初始化中');
    case LoadState.loading:
      print('加载中');
    case LoadState.ready:
      print('就绪');
    case LoadState.error:
      print('错误: ${engine.error}');
  }
});

// 监听加载进度
engine.onProgress.listen((progress) {
  print('进度: $progress');
});
```

## API 参考

### LlamaEngine

核心推理引擎，管理模型加载和文本生成。

```dart
class LlamaEngine {
  // 状态
  LoadState get state;
  String? get error;
  String? get modelPath;
  bool get isReady;
  bool get isGenerating;

  // 事件流
  Stream<LoadState> get onStateChange;
  Stream<LoadProgress> get onProgress;

  // 方法
  Future<bool> load(String modelPath);
  Future<List<int>> tokenize(String text, {bool addBos = false});
  Future<String> applyChatTemplate(List<Map<String, String>> messages);
  Stream<TokenGeneration> generate(List<int> tokens, {int maxTokens = 1024});
  void stop();
  Future<void> reset();
  Future<void> dispose();
}
```

### LlamaChat

高级对话接口，自动管理对话历史和模板。

```dart
class LlamaChat {
  LlamaChat({
    required LlamaEngine engine,
    String systemPrompt = '',
    int maxTokens = 1024,
    Future<String> Function(String conversationText)? summarizeCallback,
  });

  List<LlamaChatMessage> get history;
  Stream<LlamaChatMessage> get onMessage;
  bool get isReady;
  bool get isGenerating;
  String get conversationSummary;  // 当前对话摘要

  void clearHistory();
  void stop();
  Stream<String> sendMessage(String userMessage);
  Future<String> sendMessageAndWait(String userMessage);
  Future<void> reset();
  void dispose();
}
```

### LlamaChatMessage

对话消息模型。

```dart
enum LlamaMessageRole { system, user, assistant }

class LlamaChatMessage {
  final LlamaMessageRole role;
  final String content;

  const LlamaChatMessage({required role, required content});
  const LlamaChatMessage.system(String content);
  const LlamaChatMessage.user(String content);
  const LlamaChatMessage.assistant(String content);

  Map<String, String> toMap();
}
```

### TokenGeneration

生成结果，包含完整的 token 信息。

```dart
class TokenGeneration {
  final int token;      // Token ID
  final String text;    // Token 文本
  final bool isEnd;     // 是否为结束 token
}
```

### LlamaException

统一异常处理。

```dart
enum LlamaErrorType {
  model, tokenize, context,
  session, inference, backend, kvCache,
}

class LlamaException implements Exception {
  final LlamaErrorType type;
  final String message;
  final Map<String, dynamic>? details;

  // 工厂方法
  factory LlamaException.model(String message, {String? filePath});
  factory LlamaException.tokenize(String message, {String? text});
  factory LlamaException.context(String message);
  factory LlamaException.inference(String message, {int? tokenIndex});
  // ...
}
```

## 智能上下文压缩

当对话超过 context window 限制时，LlamaChat 支持自动压缩历史对话，而不是简单丢弃。

### 工作原理

```
Context 溢出检测 → 生成摘要 → KV Cache 重建
     ↓                ↓              ↓
  _nPast + tokens  summarizeCallback  清空 KV
  > nCtx           (用户自定义)       ↓
                                    System Prompt
                                    + 摘要
                                    + 最近对话
```

### 使用方式

```dart
final chat = LlamaChat(
  engine: engine,
  systemPrompt: '你是一个 AI 助手',
  summarizeCallback: (conversationText) async {
    // 方案 1: 使用新的 LlamaEngine 生成摘要
    final summaryEngine = LlamaEngine();
    await summaryEngine.load(modelPath);
    
    final prompt = '请总结以下对话，保留关键信息，不超过200字：\n$conversationText';
    final tokens = await summaryEngine.tokenize(prompt, addBos: false);
    
    final buffer = StringBuffer();
    await for (final gen in summaryEngine.generate(tokens, maxTokens: 256)) {
      buffer.write(gen.text);
      if (gen.isEnd) break;
    }
    
    await summaryEngine.dispose();
    return buffer.toString();
    
    // 方案 2: 使用外部 API (如 OpenAI)
    // return await openAI.summarize(conversationText);
    
    // 方案 3: 简单截断
    // return conversationText.length > 500 
    //     ? conversationText.substring(0, 500) 
    //     : conversationText;
  },
);

// 查看当前摘要
print('摘要: ${chat.conversationSummary}');
```

### KV Cache 重建流程

1. **检测溢出**: `_nPast + newTokens > nCtx`
2. **请求摘要**: 调用 `summarizeCallback(conversationText)`
3. **清空 KV Cache**: `llama_memory_clear` + `llama_synchronize`
4. **重建顺序**:
   - System Prompt tokens (始终保留)
   - 摘要 tokens (`对话历史摘要：{summary}\n\n`)
   - 最近对话 tokens (尽可能多)

### 优势

| 特性 | 说明 |
|------|------|
| 保留 System Prompt | 模型始终遵循初始指令 |
| 压缩历史 | 摘要保留关键信息，节省 context 空间 |
| 保留最近对话 | 最新交互不会丢失 |
| 自定义摘要 | 用户可选择摘要生成方式 |
| M-RoPE 兼容 | 正确处理位置编码，避免崩溃 |

## 构建说明

### 预编译库

本项目使用 **Native Assets** 机制，构建时自动从 GitHub Releases 下载预编译的原生库：

- **macOS**: ARM64, x86_64 (Metal 加速)
- **iOS**: Device (ARM64), Simulator (ARM64, x86_64)
- **Android**: ARM64, x86_64
- **Linux**: x86_64
- **Windows**: x86_64

无需本地编译 llama.cpp。

### 版本管理

在 `pubspec.yaml` 中指定 llama.cpp 版本：

```yaml
llama_cpp_version: b8369
```

或通过环境变量：

```bash
export LLAMA_CPP_VERSION=b8369
```

### 运行示例

```bash
cd example
flutter pub get
flutter run -d macos
```

## 项目结构

```
lib/
├── llama_engine.dart          # 引擎 API
├── llama_chat.dart            # 对话 API
├── llama_chat_message.dart    # 消息模型
└── src/
    ├── engine/
    │   ├── backend/           # 后端管理
    │   ├── model/             # 模型加载
    │   ├── context/           # 推理上下文
    │   ├── tokenizer/         # Tokenizer
    │   ├── cache/             # KV Cache
    │   ├── sampling/          # 采样配置
    │   └── exceptions/        # 异常定义
    ├── llama_isolate.dart     # Isolate 封装
    └── utils/                 # 工具类

hook/
└── build.dart                 # Native Assets 构建脚本
```

## 注意事项

- 模型文件需为 **GGUF 格式**
- 移动端建议使用量化模型（Q4_K_M 或更低精度）
- 确保有足够的内存（模型大小 × 1.2）
- iOS/macOS 支持 Metal GPU 加速
- Android 支持 Vulkan GPU 加速（需设备支持）

## License

MIT License
