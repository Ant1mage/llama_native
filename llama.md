# 🏗️ llama.cpp 全平台工业级封装架构 (Dart FFI 增强版)

本架构设计遵循单一职责原则（SRP），旨在支持从移动端（iOS/Android）到桌面端（macOS/Windows/Linux）的全平台部署，并深度集成动态推理参数控制。

---

## 1. 核心模块类定义 (Core Module Classes)

| 类名 | 职责 (Responsibility) | 核心动态参数 / 属性 |
| :--- | :--- | :--- |
| **`LlamaBackend`** | **全局硬件加速控制器**。负责探测并初始化各平台后端：Metal (Apple), Vulkan (Android/Linux), CUDA (Nvidia), 或 CPU (Generic)。 | `gpu_layers`, `use_mmap`, `use_mlock` |
| **`LlamaModel`** | **静态资源管理器**。负责 GGUF 权重加载。针对全平台优化内存映射（mmap），确保大模型在移动端不闪退。 | `model_path`, `vocab_only`, `split_mode` |
| **`LlamaContext`** | **推理执行引擎**。管理 KV Cache 缓冲区。负责执行具体的 `decode` 任务。 | `n_ctx` (上下文长度), `n_batch`, `n_threads` |
| **`LlamaBatch`** | **数据流水线**。动态构建计算批次。工业级实现应支持“持续批处理”（Continuous Batching）。 | `n_tokens`, `logits_all` |
| **`SamplingChain`** | **动态采样策略库**。核心调优层，负责根据用户输入的动态参数筛选最终 Token。 | `temp`, `top_p`, `min_p`, `penalty` |
| **`KVCacheManager`** | **长文本策略官**。负责上下文滚动（Sliding Window）与序列平移，防止 OOM。 | `n_past`, `n_remain`, `keep_prefix` |
| **`ChatTokenizer`** | **协议适配器**。自动识别并应用各模型官方的 Chat Template（如 Llama3, Qwen, Mistral）。 | `add_bos`, `parse_special_tokens` |
| **`SessionState`** | **持久化快照**。将会话内存序列化，支持跨平台同步与瞬间恢复对话。 | `state_id`, `buffer_size` |

---

## 2. 推理生命周期与动态参数控制



### A. 全平台环境初始化
1. **Backend Select**: 脚本自动检测环境。在 macOS/iOS 开启 Metal，在 Android 尝试 Vulkan，在 Windows 优先选择 CUDA/DirectCompute。
2. **Model Loading**: 动态设置 `n_gpu_layers`。比如在 M1 上可设为全量（100层），在低端手机上则部分卸载到 CPU。

### B. 动态采样参数 (Runtime Sampling)
每次生成任务均可注入 `InferenceConfig`，由 `SamplingChain` 实时应用：
* **Temperature (温度)**: 控制随机性。
* **Top-P / Min-P**: 过滤概率长尾，保证逻辑连贯性。
* **Repetition Penalty**: 解决模型“复读机”问题。
* **Logit Bias**: 动态干预特定词汇出现的概率（如强制模型不输出某些禁词）。

### C. 智能上下文管理 (Smart Context Control)
针对全平台（尤其是内存受限的移动端）：
1. **自动截断 (Auto-Truncation)**: 当 `n_past > n_ctx` 时，`KVCacheManager` 自动触发。
2. **保护区逻辑**: 强制保留 System Prompt 区域（Token 0~N），确保模型角色设定不丢失。
3. **滑动窗口 (Sliding Window)**: 像处理视频流一样处理 Token，仅保留最新的上下文，显著降低显存压力。

---

## 3. 全平台适配技术栈 (Platform Stack)

| 平台 | 硬件加速后端 | 内存策略 |
| :--- | :--- | :--- |
| **macOS (M-Series)** | **Metal** | 统一内存管理，支持大上下文。 |
| **iOS** | **Metal** | 严苛的 `Memory Limit` 监控，需动态调整 `n_ctx`。 |
| **Android** | **Vulkan / CLBlast** | 异构计算适配，重点优化 NPU/GPU 协作。 |
| **Windows** | **CUDA / Vulkan** | 优先 Nvidia 显存分配，备选系统内存。 |
| **Linux** | **CUDA / CPU** | 灵活的计算集群适配。 |

---

## 4. 关键 FFI 开发规范 (FFI Constraints)

1. **非阻塞异步 (Async Isolate)**: 不要直接在 Flutter 主 Isolate 调用 FFI 推理。必须在 Rust 侧使用独立线程或 Dart `compute` 开启后台 Isolate，通过 `Stream` 异步回传 Token。
2. **零拷贝通信 (Zero-copy)**: 采样后的 Token ID 直接在 C 内存空间转换，仅将最终生成的 String 拷贝回 Dart 堆，降低内存抖动。
3. **优雅降级 (Fallback)**: 如果硬件加速初始化失败，系统应能无缝切换至 `GGML_TYPE_F16` 的 CPU 纯指令集推理。