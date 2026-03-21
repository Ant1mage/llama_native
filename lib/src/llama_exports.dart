/// Llama Native - 全平台工业级封装
///
/// 提供易用、高效、全平台兼容的 Dart API for llama.cpp
library llama_exports;

// High-level API
export 'package:llama_native/llama_engine.dart' show LlamaEngine, LoadState, LoadProgress;
export 'package:llama_native/llama_chat.dart' show LlamaChat, LlamaChatMessage, LlamaMessageRole;
export 'package:llama_native/src/engine/context/token_generation.dart' show TokenGeneration, KVCacheStatus;
export 'package:llama_native/src/engine/context/performance_metrics.dart' show PerformanceMetrics;
export 'package:llama_native/src/engine/exceptions/llama_exceptions.dart' show LlamaException, LlamaErrorType;
export 'package:llama_native/src/utils/platform_info.dart' show PlatformInfo;
export 'package:llama_native/src/engine/cache/kv_cache_manager.dart'
    show KVCacheThresholdCallback, KVCacheResetCallback;
