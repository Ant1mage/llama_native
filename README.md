# llama_native

Flutter FFI bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp) - 端云协同推理引擎

## 特性

### 核心功能（端云协同必备）

| 功能 | 说明 |
|------|------|
| 🗺️ **mmap 内存映射** | 按需加载模型，减少内存占用 |
| 🔄 **KV Cache 滚动管理** | Context Shifting 自动保留 System Prompt |
| ⚡ **Metal 加速** | iOS/macOS 原生 GPU 加速 |
| 📝 **统一 Tokenization** | BPE/SPM 分词，端云 Token 计数一致 |
| 💬 **Chat Template** | 自动检测模型模板格式 |
| 🎲 **采样链** | Temperature/Top-P/Min-P/Repeat Penalty |
| 📊 **Embedding 提取** | 语义向量，支持端云路由决策 |
| ⏱️ **性能指标** | 实时 ms/tok 监控，过载自动告警 |
| ⏹️ **异步中断** | 随时停止生成，省电省时 |
| 🧵 **线程控制** | 根据设备核心数自动配置 |

### 高级功能

- 🧠 **智能上下文压缩**: Context 溢出时自动生成摘要
- 📦 **会话持久化**: KV Cache 快照保存与恢复
- 🎯 **Grammar 约束**: JSON Schema 输出格式化
- 🔧 **Function Calling**: 工具调用支持

## 快速开始

### 1. 添加依赖

```yaml
dependencies:
  llama_native: ^0.0.1
```

### 2. 基础使用

```dart
import 'package:llama_native/llama_native.dart';

final engine = LlamaEngine();

if (await engine.load('path/to/model.gguf')) {
  final tokens = await engine.tokenize('你好');
  await for (final gen in engine.generate(tokens, maxTokens: 256)) {
    print(gen.text);
    if (gen.isEnd) break;
  }
}

await engine.dispose();
```

### 3. 对话示例

```dart
final engine = LlamaEngine();
await engine.load('path/to/model.gguf');

final chat = LlamaChat(
  engine: engine,
  systemPrompt: '你是一个 AI 助手',
  maxTokens: 1024,
);

await for (final text in chat.sendMessage('你好')) {
  stdout.write(text);
}

chat.dispose();
await engine.dispose();
```

### 4. 性能监控

```dart
final engine = LlamaEngine();
await engine.load('path/to/model.gguf');

// 获取性能指标
final metrics = engine.performanceMetrics;
print('ms/tok: ${metrics.msPerToken}');
print('t/s: ${metrics.tokensPerSecond}');

// 检测设备负载
if (metrics.isOverloaded) {
  print('设备过载，建议切换云端');
}
```

### 5. Embedding 提取

```dart
import 'package:llama_native/src/engine/embeddings/embeddings.dart';

final embeddings = await LlamaEmbeddings.create(model);

final vec1 = embeddings.embed('你好世界');
final vec2 = embeddings.embed('Hello World');

final similarity = vec1.similarity(vec2);
print('相似度: $similarity');

// 用于端云路由决策
if (similarity > 0.8) {
  // 本地处理
} else {
  // 云端处理
}
```

## API 参考

### LlamaEngine

```dart
class LlamaEngine {
  LoadState get state;
  bool get isReady;
  bool get isGenerating;
  PerformanceMetrics get performanceMetrics;

  Future<bool> load(String modelPath);
  Future<List<int>> tokenize(String text, {bool addBos = false});
  Stream<TokenGeneration> generate(List<int> tokens, {int maxTokens = 1024});
  void stop();
  Future<void> reset();
  Future<void> dispose();
}
```

### LlamaChat

```dart
class LlamaChat {
  List<LlamaChatMessage> get history;
  bool get isGenerating;
  String get conversationSummary;

  Stream<String> sendMessage(String userMessage);
  Future<String> sendMessageAndWait(String userMessage);
  void stop();
  void clearHistory();
  Future<void> reset();
}
```

### PerformanceMetrics

```dart
class PerformanceMetrics {
  double get msPerToken;        // 每个 token 耗时
  double get tokensPerSecond;   // 每秒生成 token 数
  bool get isOverloaded;        // 设备是否过载 (ms/tok > 200)
  bool get isSlowDevice;        // 是否为慢速设备 (t/s < 5)
}
```

### SamplingConfig

```dart
final config = SamplingConfig(
  temperature: 0.8,
  topP: 0.95,
  minP: 0.05,
  topK: 40,
  penaltyRepeat: 1.1,
  frequencyPenalty: 0.0,
  presencePenalty: 0.0,
);
```

## 端云协同架构

```
┌─────────────────────────────────────────────────────────────┐
│                    端云协同决策流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户输入 ──→ Embedding 提取 ──→ 语义路由决策               │
│                                    │                        │
│                    ┌───────────────┴───────────────┐        │
│                    ▼                               ▼        │
│              本地推理                        云端 API        │
│         (LlamaEngine)                  (OpenAI/等)          │
│                    │                               │        │
│                    ▼                               ▼        │
│              性能监控                        响应处理        │
│         (ms/tok 检测)                                           │
│                    │                                          │
│         过载时切换 ───────────────────────────→ 云端          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 路由决策示例

```dart
class HybridRouter {
  final LlamaEngine localEngine;
  final CloudProvider cloudProvider;
  
  Future<String> generate(String prompt) async {
    final metrics = localEngine.performanceMetrics;
    
    if (metrics.isOverloaded) {
      return cloudProvider.complete(prompt);
    }
    
    final embedding = await localEmbeddings.embed(prompt);
    if (needsCloudProcessing(embedding)) {
      return cloudProvider.complete(prompt);
    }
    
    return localGenerate(prompt);
  }
}
```

## 智能上下文压缩

```dart
final chat = LlamaChat(
  engine: engine,
  systemPrompt: '你是一个 AI 助手',
  summarizeCallback: (text) async {
    return await generateSummary(text);
  },
);
```

## 项目结构

```
lib/
├── llama_engine.dart          # 引擎 API
├── llama_chat.dart            # 对话 API
└── src/
    ├── engine/
    │   ├── backend/           # 后端管理 (mmap, Metal)
    │   ├── model/             # 模型加载
    │   ├── context/           # 推理上下文 + 性能指标
    │   ├── tokenizer/         # Tokenizer + Chat Template
    │   ├── cache/             # KV Cache 滚动管理
    │   ├── sampling/          # 采样配置
    │   ├── embeddings/        # Embedding 提取
    │   └── exceptions/        # 统一异常
    └── llama_isolate.dart     # Isolate 封装
```

## 注意事项

- 模型文件需为 **GGUF 格式**
- 移动端建议使用量化模型（Q4_K_M 或更低精度）
- 确保有足够的内存（模型大小 × 1.5）

## License

MIT License
