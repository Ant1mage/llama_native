# Llama Native

Flutter FFI 插件，提供 llama.cpp 的全平台 Dart 封装，支持本地大模型推理。

## 特性

- 🚀 **全平台支持**: Android, iOS, macOS, Linux, Windows
- 📦 **Native Assets**: 自动下载预编译库，无需本地编译
- 💬 **对话支持**: ChatTokenizer 和多种 Chat Template
- ⚡ **流式生成**: 实时 token 输出，不阻塞 UI
- 🎯 **Grammar 约束**: JSON Schema 转 GBNF，结构化输出
- 🔧 **Function Calling**: 支持工具调用和函数执行
- 🗂️ **KV Cache 管理**: 会话状态保存与恢复

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

// 智能配置（根据硬件自动优化）
final config = InferenceConfig.defaults();

// 或针对特定模型优化
final config = InferenceConfig.forModel(
  modelSizeMB: 4000,  // 模型大小 MB
  modelLayers: 32,    // 模型层数
);

// 或手动指定参数
final config = InferenceConfig.defaults(
  nCtx: 8192,
  nGpuLayers: 40,
  sampling: SamplingConfig(temperature: 0.7, topP: 0.9),
);

// 创建上下文
final context = LlamaContext.create(model, config);

// 流式生成
final tokens = model.tokenize('你好', addBos: true);
final stream = context.generateStream(tokens, maxTokens: 256);
await for (final result in stream) {
  print(result.text);
}

// 清理资源
context.dispose();
model.dispose();
backend.dispose();
```

### 3. 查看硬件信息

```dart
// 打印当前硬件信息
print(PlatformInfo.getHardwareInfo());

// 获取推荐配置
print('推荐 GPU 层数: ${PlatformInfo.recommendedGpuLayers()}');
print('推荐上下文长度: ${PlatformInfo.recommendedContextLength()}');
print('推荐批次大小: ${PlatformInfo.recommendedBatchSize()}');
print('系统内存: ${PlatformInfo.systemMemoryMB}MB');
print('可用显存: ${PlatformInfo.availableVRAMMB}MB');
```

### 4. 对话示例

```dart
// 初始化 ChatTokenizer
final tokenizer = ChatTokenizer(
  model: model,
  templateType: ChatTemplateType.chatml,
);

// 构建对话
final messages = [
  const ChatMessage.system('你是一个 AI 助手'),
  const ChatMessage.user('你好'),
];

// 应用模板并 tokenize
final prompt = tokenizer.applyTemplate(messages);
final tokens = model.tokenize(prompt, addBos: false);

// 生成回复
final stream = context.generateStream(tokens, maxTokens: 512);
final buffer = StringBuffer();
await for (final result in stream) {
  buffer.write(result.text);
  if (result.isEnd) break;
}
print(buffer.toString());
```

### 5. JSON 结构化输出

```dart
// 定义 JSON Schema
final schema = {
  'type': 'object',
  'properties': {
    'name': {'type': 'string'},
    'age': {'type': 'integer'},
  },
  'required': ['name', 'age'],
};

// 创建 Grammar
final grammar = Grammar.fromJsonSchema(schema);

// 使用 Grammar 生成
final stream = context.generateStream(tokens, maxTokens: 256, grammar: grammar);
```

### 6. Function Calling

```dart
// 定义函数
final functions = [
  FunctionDefinition(
    name: 'get_weather',
    description: '获取指定城市的天气',
    parameters: {
      'type': 'object',
      'properties': {
        'city': {'type': 'string', 'description': '城市名称'},
      },
      'required': ['city'],
    },
  ),
];

// 创建 Function Manager
final functionManager = FunctionManager(functions: functions);

// 生成并解析函数调用
final grammar = functionManager.createGrammar();
final stream = context.generateStream(tokens, grammar: grammar);
```

## 核心模块

| 模块 | 说明 |
|------|------|
| `LlamaBackend` | 后端初始化，硬件加速检测 |
| `LlamaModel` | 模型加载，tokenize/detokenize |
| `LlamaContext` | 推理引擎，流式生成 |
| `InferenceConfig` | 推理参数配置 |
| `ChatTokenizer` | Chat Template 处理 |
| `KVCacheManager` | KV Cache 管理与快照 |
| `SessionManager` | 会话状态管理 |
| `SamplingConfig` | 采样策略配置 |
| `Grammar` | GBNF 语法约束 |
| `FunctionManager` | Function Calling 管理 |

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
    ├── grammar/               # Grammar 约束
    ├── function/              # Function Calling
    └── utils/                 # 工具类

hook/
└── build.dart                 # Native Assets 构建脚本

.github/workflows/
└── build-dylib.yml            # CI/CD 预编译库构建
```

## 支持的 Chat Template

| Template | 模型 |
|----------|------|
| `chatml` | Qwen, Yi, 等 |
| `llama3` | Llama 3.x |
| `qwen` | Qwen 系列 |
| `mistral` | Mistral, Mixtral |
| `alpaca` | Alpaca 系列 |

## 注意事项

- 模型文件需为 **GGUF 格式**
- 移动端建议使用量化模型（Q4_K_M 或更低精度）
- 确保有足够的内存（模型大小 × 1.2）
- iOS/macOS 支持 Metal GPU 加速
- Android 支持 Vulkan GPU 加速（需设备支持）

## License

MIT License
