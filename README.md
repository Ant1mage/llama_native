# llama_native

Flutter FFI bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp) - 端云协同推理引擎

## 特性

### 核心功能（端云协同必备）

| 功能 | 说明 |
|------|------|
| 🗺️ **mmap 内存映射** | 按需加载模型，减少内存占用 |
| 🔄 **KV Cache 管理** | MoE 兼容，物理状态监测，自动截断 |
| ⚡ **Metal 加速** | iOS/macOS 原生 GPU 加速 |
| 📝 **统一 Tokenization** | BPE/SPM 分词，端云 Token 计数一致 |
| 💬 **Chat Template** | 自动检测模型模板格式，支持 Gemma/MiniCPM |
| 🎲 **采样链** | Temperature/Top-P/Min-P/Repeat Penalty/Mirostat |
| 📊 **Embedding 提取** | 语义向量，支持端云路由决策 |
| ⏱️ **性能指标** | 实时 ms/tok 监控，过载自动告警 |
| ⏹️ **异步中断** | 随时停止生成，省电省时 |
| 🧵 **线程控制** | 根据设备核心数自动配置 |

### 性能优化（显存/内存降低 50%+）

| 优化项 | 效果 | 适用场景 |
|--------|------|----------|
| ⚡ **Flash Attention** | 显存降低 30-50% | 长上下文推理 |
| 🗜️ **KV Cache 量化** | KV 显存再降 50% | 内存受限设备 |
| 📦 **Q8_0 量化** | 模型体积/显存大幅减少 | 通用场景 |

> 默认配置已启用 Flash Attention + Q8_0 KV 量化，兼顾性能与质量

### 自定义优化配置

如需调整优化级别，可手动配置 `LlamaBackendConfig`：

```dart
import 'package:llama_native/src/llama_native_bindings.dart' as bindings;

// 仅启用 Flash Attention（推荐移动设备）
const config1 = LlamaBackendConfig(
  flashAttention: FlashAttentionType.auto,
  kvCacheTypeK: null,  // 不量化
  kvCacheTypeV: null,
);

// KV Cache INT8 量化（节省显存）
const config2 = LlamaBackendConfig(
  flashAttention: FlashAttentionType.auto,
  kvCacheTypeK: bindings.ggml_type.GGML_TYPE_Q8_0,
  kvCacheTypeV: bindings.ggml_type.GGML_TYPE_Q8_0,
);

// 关闭所有优化（调试用）
const config3 = LlamaBackendConfig(
  flashAttention: FlashAttentionType.disabled,
);
```

### 显存占用预估

| 模型 + 精度 | 原始 | + Flash Attn | + KV Q8_0 |
|-------------|------|--------------|-----------|
| 7B Q4_K_M | ~4 GB | ~2.5 GB | ~2 GB |
| 13B Q4_K_M | ~7 GB | ~4.5 GB | ~3.5 GB |
| 7B Q8_0 | ~7 GB | ~4 GB | ~3 GB |

> 实测 Apple Silicon M2 Pro (24GB) 可流畅运行 13B Q4_K_M 模型

### 高级功能

- 🧠 **MoE 兼容缓存**: 20% 专家缓冲区锁定，物理位置断层处理
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
    print('KV: ${gen.kvUsed}/${gen.kvTotal} (${gen.kvUsagePercent.toStringAsFixed(1)}%)');
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

### 4. 性能监控与 KV 状态

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

// 生成时监控 KV 缓存状态
await for (final gen in engine.generate(tokens, maxTokens: 256)) {
  if (gen.kvUsagePercent > 80) {
    print('KV 缓存接近安全阈值');
  }
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

  Stream<String> sendMessage(String userMessage);
  Future<String> sendMessageAndWait(String userMessage);
  void stop();
  void clearHistory();
  Future<void> reset();
}
```

### TokenGeneration

```dart
class TokenGeneration {
  final int token;              // Token ID
  final String text;            // 解码文本
  final bool isEnd;             // 是否结束
  
  // KV 缓存状态
  final int kvUsed;             // 已用槽位
  final int kvTotal;            // 总容量
  final double kvUsagePercent;  // 使用百分比
}
```

### KVCacheStatus

```dart
class KVCacheStatus {
  final int used;               // 已用槽位
  final int total;              // 总容量
  final int safeCapacity;       // 安全阈值 (80%)
  final int moeBufferSlots;     // MoE 缓冲区 (20%)
  final double usagePercent;    // 使用百分比
  final bool isOverSafeThreshold;  // 是否超过安全阈值
  final bool isFull;            // 是否满载
  
  bool get needsReset;          // 是否需要重置
  int get remaining;            // 剩余槽位
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
  useMirostat: true,           // Mirostat 采样
  mirostatTau: 5.0,
  mirostatEta: 0.1,
);
```

## KV 缓存管理策略

### MoE 兼容设计

```
┌─────────────────────────────────────────────────────────────┐
│                    KV 缓存水位控制                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  0% ────────────────────────────────────────────── 80%      │
│  │                    可用区域                    │ MoE缓冲 │
│  │                                               │  20%    │
│  │                                               │         │
│  │  自动截断 (75%)                                │         │
│  │      ↓                                        │         │
│  │  紧急截断 (80%)                                │         │
│  │      ↓                                        │         │
│  │  满载异常 (100%) ← 硬拦截，终止推理             │         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 水位阈值

| 阈值 | 行为 |
|------|------|
| 75% | 自动截断中间历史，保留 System Prompt + 最近对话 |
| 80% | 紧急截断或请求重建 |
| 100% | 抛出异常终止推理，等待外部重置 |

### 物理状态监测

```dart
// 生成循环中实时监测
await for (final gen in engine.generate(tokens, maxTokens: 1024)) {
  // 检测 KV 缓存状态
  if (gen.kvUsagePercent > 75) {
    print('警告: KV 缓存接近阈值');
  }
  
  // 满载时自动中断
  if (gen.kvUsed >= gen.kvTotal) {
    print('KV 缓存满载，需要重置');
    break;
  }
}
```

### 全量重置

```dart
// 物理级重置 KV 缓存
await engine.reset();

// 或在 isolate 中
// context.fullResetKVCache();
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
│         (ms/tok + KV状态)                                    │
│                    │                                          │
│         过载/KV满载 ───────────────────────→ 云端            │
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
    
    // 性能过载切换云端
    if (metrics.isOverloaded) {
      return cloudProvider.complete(prompt);
    }
    
    // 语义路由
    final embedding = await localEmbeddings.embed(prompt);
    if (needsCloudProcessing(embedding)) {
      return cloudProvider.complete(prompt);
    }
    
    return localGenerate(prompt);
  }
  
  Stream<String> localGenerate(String prompt) async* {
    await for (final gen in localEngine.generate(tokens)) {
      // KV 缓存满载时切换云端
      if (gen.kvUsagePercent > 80) {
        yield* cloudProvider.stream(prompt);
        return;
      }
      yield gen.text;
    }
  }
}
```

## 项目结构

```
lib/
├── llama_engine.dart          # 引擎 API
├── llama_chat.dart            # 对话 API
└── src/
    ├── engine/
    │   ├── backend/           # 后端管理 (mmap, Metal)
    │   ├── model/             # 模型加载 + Embedding
    │   ├── context/           # 推理上下文 + 性能指标 + KV状态
    │   ├── tokenizer/         # Tokenizer + Chat Template
    │   ├── cache/             # KV Cache 管理 (MoE兼容)
    │   ├── sampling/          # 采样配置
    │   ├── embeddings/        # Embedding 提取
    │   └── exceptions/        # 统一异常
    └── llama_isolate.dart     # Isolate 封装
```

## 注意事项

- 模型文件需为 **GGUF 格式**
- 移动端建议使用量化模型（Q4_K_M 或更低精度）
- 确保有足够的内存（模型大小 × 1.5）
- MoE 模型建议预留 20% KV 缓存缓冲区

## License

MIT License
