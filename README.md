# Llama Native

Flutter FFI 插件，提供 llama.cpp 的全平台 Dart 封装。

## 特性

- 🚀 **全平台支持**: Android, iOS, macOS, Linux, Windows
- 📦 **高级 API**: 简洁易用的 Dart 接口
- 🔧 **CMake 构建**: 自动编译 llama.cpp
- 💬 **对话支持**: ChatTokenizer 和会话管理
- ⚡ **流式生成**: 实时 token 输出

## 快速开始

### 1. 添加依赖

```yaml
dependencies:
  llama_native: ^0.0.1
```

### 2. 基础使用

```dart
import 'package:llama_native/llama_native.dart';

// 初始化后端
final backend = LlamaBackend.instance;
await backend.initialize();

// 加载模型
final model = LlamaModel.load(
  LlamaModelConfig(modelPath: 'path/to/model.gguf')
);

// 创建推理配置
final config = InferenceConfig.defaultMacOS();

// 创建上下文
final context = LlamaContext.create(model, config);

// 流式生成
final stream = context.generateStream(tokens, maxTokens: 256);
await for (final result in stream) {
  print(result.text);
}

// 清理资源
context.dispose();
model.dispose();
backend.dispose();
```

### 3. 对话示例

```dart
// 初始化 ChatTokenizer
final tokenizer = ChatTokenizer(
  model: model,
  templateType: ChatTemplateType.qwen,
  addBos: true,
);

// 构建对话
final messages = [
  const ChatMessage.system('你是一个 AI 助手'),
  const ChatMessage.user('你好'),
];

// Tokenize
final tokens = tokenizer.tokenizeMessages(messages);

// 生成回复
final response = await context.generate(tokens);
```

## 核心模块

| 模块 | 说明 |
|------|------|
| `LlamaBackend` | 后端初始化管理 |
| `LlamaModel` | 模型加载与配置 |
| `LlamaContext` | 推理执行引擎 |
| `InferenceConfig` | 推理参数配置 |
| `ChatTokenizer` | Chat Template 处理 |
| `KVCacheManager` | KV Cache 管理 |
| `SessionManager` | 会话状态管理 |
| `SamplingConfig` | 采样策略配置 |

## 构建说明

### 前置要求

- Flutter SDK >= 3.3.0
- Dart SDK >= 3.11.1
- CMake 3.22+
- C++ 编译器

### 编译 llama.cpp

```bash
# 构建原生库
dart run hook/build.dart
```

### 运行示例

```bash
cd example
flutter pub get
dart run lib/main.dart
```

## 项目结构

```
lib/
├── llama_native.dart          # 主入口
├── llama_native_bindings.dart # FFI 绑定
└── src/
    ├── backend/               # 后端管理
    ├── model/                 # 模型加载
    ├── context/               # 推理上下文
    ├── tokenizer/             # Tokenizer
    ├── cache/                 # KV Cache
    ├── session/               # 会话管理
    ├── sampling/              # 采样配置
    └── utils/                 # 工具类
```

## 注意事项

- 模型文件需为 GGUF 格式
- 首次运行需要编译 llama.cpp（可能需要几分钟）
- 移动端建议量化模型（Q4_K_M 或更低精度）
- 确保有足够的内存（模型大小 × 1.2）

## License

MIT License
