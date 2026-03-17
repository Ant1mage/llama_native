/// Llama Native - 全平台工业级封装
///
/// 提供易用、高效、全平台兼容的 Dart API for llama.cpp
library exports;

// High-level API (推荐使用)
export 'package:llama_native/llama_engine.dart' show LlamaEngine, LoadState, LoadProgress;
export 'package:llama_native/llama_chat.dart' show LlamaChat, ChatMessage, MessageRole;

// Platform Info
export 'package:llama_native/src/utils/platform_info.dart' show PlatformInfo, HardwareAcceleration;

// Sampling
export 'package:llama_native/src/engine/sampling/sampling_config.dart' show SamplingConfig, LogitBias;

// Logging
export 'package:llama_native/src/log/logger.dart' show Logger;

// Isolate (低级 API)
export 'package:llama_native/src/engine/isolate/llama_isolate.dart' show LlamaIsolate, LlamaIsolateConfig;
