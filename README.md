# llama_native

Flutter FFI bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp) - 端云协同推理引擎

## 特性

### 核心功能

| 功能 | 说明 |
|------|------|
| 🗺️ **mmap 内存映射** | 按需加载模型，减少内存占用 |
| 🔄 **KV Cache 管理** | 阈值监控、自动截断、上下文注入 |
| ⚡ **Metal 加速** | iOS/macOS 原生 GPU 加速 |
| 📝 **统一 Tokenization** | BPE/SPM 分词，端云 Token 计数一致 |
| 💬 **Chat Template** | 自动检测模型模板格式 |
| 🎲 **采样链** | Temperature/Top-P/Min-P/Repeat Penalty/Mirostat |
| 📊 **Embedding 提取** | 语义向量，支持端云路由决策 |
| ⏱️ **性能指标** | 实时 ms/tok 监控，过载自动告警 |
| ⏹️ **异步中断** | 随时停止生成，省电省时 |

### KV Cache 联动（新）

| 功能 | 说明 |
|------|------|
| **阈值回调** | 75%/80%/100% 三级阈值触发回调 |
| **暂停/恢复** | 支持暂停 KV Cache 管理，等待外部处理 |
| **上下文注入** | 将摘要/快照注入 KV Cache，保持对话连续性 |

### 性能优化

| 优化项 | 效果 | 适用场景 |
|--------|------|----------|
| ⚡ **Flash Attention** | 显存降低 30-50% | 长上下文推理 |
| 🗜️ **KV Cache 量化** | KV 显存再降 50% | 内存受限设备 |
| 📦 **Q8_0 量化** | 模型体积/显存大幅减少 | 通用场景 |

---

## 快速开始

### 基础使用

```dart
import 'package:llama_native/llama_native.dart';

final engine = LlamaEngine();

if (await engine.load('path/to/model.gguf')) {
  final tokens = await engine.tokenize('你好');
  await for (final gen in engine.generate(tokens, maxTokens: 256)) {
    print(gen.text);
    print('KV: ${gen.kvUsagePercent.toStringAsFixed(1)}%');
    if (gen.isEnd) break;
  }
}

await engine.dispose();
```

### 对话示例

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
```

### KV Cache 回调与注入

```dart
final engine = LlamaEngine();
await engine.load('path/to/model.gguf');

// 设置 KV Cache 回调
engine.setKVCacheCallbacks(
  onNearThreshold: (usage) {
    print('KV Cache 接近阈值: $usage%');
    // 触发记忆压缩...
  },
  onEmergency: (usage) {
    print('KV Cache 紧急: $usage%');
  },
  onFullReset: () {
    print('KV Cache 已重置');
  },
);

// 注入上下文（如记忆快照）
await engine.injectContextText('【记忆快照】用户正在开发 AI 项目...');

// 或注入 Token
final tokens = await engine.tokenize('摘要内容...');
await engine.injectContextTokens(tokens);
```

---

## API 参考

### LlamaEngine

| 方法/属性 | 说明 |
|-----------|------|
| `load(modelPath)` | 加载模型 |
| `tokenize(text)` | 文本转 Token |
| `generate(tokens)` | 流式生成 |
| `stop()` | 停止生成 |
| `reset()` | 重置上下文 |
| `dispose()` | 释放资源 |
| `kvCacheUsagePercent` | KV Cache 使用率 |
| `performanceMetrics` | 性能指标 |

**KV Cache 相关：**

| 方法 | 说明 |
|------|------|
| `setKVCacheCallbacks(...)` | 设置阈值回调 |
| `pauseKVCache()` | 暂停 KV Cache 管理 |
| `resumeKVCache()` | 恢复 KV Cache 管理 |
| `injectContextText(text)` | 注入文本到 KV Cache |
| `injectContextTokens(tokens)` | 注入 Token 到 KV Cache |

### TokenGeneration

| 属性 | 说明 |
|------|------|
| `token` | Token ID |
| `text` | 解码文本 |
| `isEnd` | 是否结束 |
| `kvUsed` | 已用槽位 |
| `kvTotal` | 总容量 |
| `kvUsagePercent` | 使用百分比 |

### KVCacheStatus

| 属性 | 说明 |
|------|------|
| `used` | 已用槽位 |
| `total` | 总容量 |
| `safeCapacity` | 安全阈值 (80%) |
| `usagePercent` | 使用百分比 |
| `isOverSafeThreshold` | 是否超过安全阈值 |
| `isFull` | 是否满载 |

---

## KV Cache 管理策略

### 阈值控制

```
┌─────────────────────────────────────────────────────────────┐
│                    KV Cache 水位控制                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  0% ────────────────────────────────────────────── 80%      │
│  │                    可用区域                    │ MoE缓冲 │
│  │                                               │  20%    │
│                                                             │
│  75% ──▶ 触发 onNearThreshold 回调                           │
│  80% ──▶ 触发 onEmergency 回调                               │
│  100% ──▶ 触发 onFullReset 回调                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 联动流程

```
KV Cache 达到 75%
    │
    ├─▶ 触发 onNearThreshold 回调
    │
    ├─▶ 外部生成记忆快照
    │
    ├─▶ 调用 injectContextText() 注入
    │
    └─▶ KV Cache 管理恢复，继续生成
```

---

## 项目结构

```
lib/
├── llama_engine.dart          # 引擎 API
├── llama_chat.dart            # 对话 API
├── llama_chat_message.dart    # 消息类型
└── src/
    ├── engine/
    │   ├── cache/             # KV Cache 管理
    │   ├── context/           # 推理上下文
    │   ├── model/             # 模型加载
    │   ├── tokenizer/         # 分词器
    │   ├── sampling/          # 采样配置
    │   ├── grammar/           # Grammar 约束
    │   ├── embeddings/        # 嵌入向量
    │   ├── function/          # 函数调用
    │   ├── multimodal/        # 多模态（预留）
    │   ├── session/           # 会话管理（预留）
    │   └── exceptions/        # 异常定义
    ├── llama_isolate.dart     # Isolate 封装
    └── llama_exports.dart     # 导出配置
```

---

## 平台支持

| 平台 | 支持 |
|------|------|
| macOS | ✅ |
| iOS | ✅ |
| Android | ✅ |
| Windows | ✅ |
| Linux | ✅ |

## License

MIT License
