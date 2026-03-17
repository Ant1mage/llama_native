/// Llama Native - 全平台工业级封装
///
/// 提供易用、高效、全平台兼容的 Dart API for llama.cpp
library llama_native;

// Backend
export 'package:llama_native/src/backend/llama_backend.dart' show LlamaBackend;
export 'package:llama_native/src/backend/llama_backend_config.dart' show LlamaBackendConfig;
export 'package:llama_native/src/utils/platform_info.dart' show PlatformInfo, HardwareAcceleration;

// Model
export 'package:llama_native/src/model/llama_model.dart' show LlamaModel, LlamaModelConfig, LlamaModelMetadata;

// Context
export 'package:llama_native/src/context/llama_context.dart' show LlamaContext;
export 'package:llama_native/src/context/inference_config.dart' show InferenceConfig;
export 'package:llama_native/src/context/token_generation.dart' show TokenGeneration;

// Exceptions
export 'package:llama_native/src/exceptions/llama_exceptions.dart'
    show LlamaModelLoadException, LlamaTokenizeException, LlamaContextInitException, LlamaSessionStateException;

// Batch
export 'package:llama_native/src/batch/llama_batch.dart' show LlamaBatch;
export 'package:llama_native/src/batch/Llama_batch_build.dart' show BatchBuilder;

// Sampling
export 'package:llama_native/src/sampling/sampling_config.dart' show SamplingConfig, LogitBias;

// Cache
export 'package:llama_native/src/cache/kv_cache_manager.dart' show KVCacheManager;
export 'package:llama_native/src/cache/kv_cache_snapshot.dart' show KVCacheSnapshot;

// Tokenizer
export 'package:llama_native/src/tokenizer/chat_tokenizer.dart'
    show ChatTokenizer, ChatTemplateType, ChatMessage, MessageRole;

// Session
export 'package:llama_native/src/session/session_state.dart' show SessionState, SessionManager;

// Grammar
export 'package:llama_native/src/grammar/grammar.dart'
    show Grammar, GrammarConfig, GrammarParseResult, JsonSchemaBuilder;

// Function Calling
export 'package:llama_native/src/function/function_definition.dart'
    show FunctionDefinition, FunctionParameter, FunctionCallResult, FunctionCallParser;
export 'package:llama_native/src/function/function_manager.dart'
    show FunctionManager, FunctionCallingHelper;

// Logging
export 'package:llama_native/src/logging/logger.dart' show Logger;

// Isolate
export 'package:llama_native/src/isolate/llama_isolate.dart' show LlamaIsolate, LlamaIsolateConfig;
