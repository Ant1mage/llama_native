# Changelog

All notable changes to this project will be documented in this file.

## [2026-03-14] - Major Refactoring & Async Removal

### 🔧 Breaking Changes - Class Renaming (Added Llama Prefix)

#### Backend Module (Split into 4 files)
- `BackendConfig` → `LlamaBackendConfig`
- `PlatformInfo` → `LlamaPlatformInfo`
- `HardwareAcceleration` → `LlamaHardwareAcceleration`
- `LlamaBackend` → (unchanged)

**New File Structure:**
- `lib/src/backend/llama_backend.dart` - LlamaBackend class only
- `lib/src/backend/llama_backend_config.dart` - NEW: LlamaBackendConfig
- `lib/src/backend/llama_platform_info.dart` - NEW: LlamaPlatformInfo
- `lib/src/backend/hardware_acceleration.dart` - NEW: LlamaHardwareAcceleration enum

#### Context Module
- `InferenceConfig` → `LlamaInferenceConfig`
- `GenerationResult` → `LlamaGenerationResult`
- `ContextInitException` → `LlamaContextInitException`
- `KVCacheManager` → `LlamaKVCacheManager`

**File:** `lib/src/context/llama_context.dart`

#### Cache Module
- `KVCacheManager` → `LlamaKVCacheManager`
- `KVCacheSnapshot` → `LlamaKVCacheSnapshot`

**File:** `lib/src/cache/kv_cache_manager.dart`

#### Sampling Module
- `SamplingConfig` → `LlamaSamplingConfig`
- `LogitBias` → `LlamaLogitBias`

**File:** `lib/src/sampling/sampling_config.dart`

---

### 🔄 API Changes - Removed Async compute()

#### Synchronous API (Default)
- `LlamaModel.load()` changed from `async Future` to sync factory constructor
- `LlamaContext.create()` changed from `async Future` to sync factory constructor
- `generateStream()` now uses `async*` and `yield` instead of `compute()`

**Migration Guide:**
```dart
// Before (Async with compute)
final model = await LlamaModel.load(config);
final context = await LlamaContext.create(model, config);

// After (Sync)
final model = LlamaModel.load(config);
final context = LlamaContext.create(model, config);
```

**For Async Support:**
Use isolate manually if needed for heavy operations.

---

### ✨ New Features - Platform Auto-Detection

#### Dynamic Parameter Calculation

**LlamaPlatformInfo** provides intelligent defaults:
- `recommendedGpuLayers()` - Based on estimated VRAM
- `recommendedContextLength()` - Platform-specific limits
- `recommendedBatchSize()` - Memory-aware sizing
- `recommendedThreads()` - CPU core-based calculation

**VRAM Estimation:**
- Apple Silicon: 60% of unified memory
- Android: 2-4GB (conservative)
- Windows: 8GB assumed
- Linux: 12GB assumed (workstation/server)

**GPU Layers Auto-Calculation:**
```dart
if (vram >= 12 * 1024) return 100;  // Full offload
if (vram >= 8 * 1024)  return 75;   // Most layers
if (vram >= 4 * 1024)  return 50;   // Half layers
if (vram >= 2 * 1024)  return 25;   // Partial
```

#### Platform-Specific Factory Methods

**LlamaBackendConfig:**
- `defaultMacOS()` - Apple Silicon Metal optimization
- `defaultAndroid()` - Vulkan optimization
- `defaultWindows()` - CUDA optimization
- `defaultLinux()` - CUDA/Vulkan optimization
- `defaultIOS()` - Metal (conservative settings)

**LlamaInferenceConfig:**
- `defaultMacOS()` - nCtx: 4096, nBatch: 512, threads: physical cores
- `defaultAndroid()` - nCtx: 2048, nBatch: 256, threads: 2-4
- `defaultWindows()` - nCtx: 4096, nBatch: 512, threads: physical cores
- `defaultLinux()` - nCtx: 4096, nBatch: 512, threads: physical cores
- `defaultIOS()` - nCtx: 2048, nBatch: 256, threads: 2-4

---

### 📦 Export Changes

**lib/llama_native.dart** now exports:
```dart
// Backend (4 separate files)
export 'src/backend/llama_backend.dart' show LlamaBackend;
export 'src/backend/llama_backend_config.dart' show LlamaBackendConfig;
export 'src/backend/llama_platform_info.dart' show LlamaPlatformInfo;
export 'src/backend/hardware_acceleration.dart' show LlamaHardwareAcceleration;

// Context
export 'src/context/llama_context.dart'
    show LlamaContext, LlamaInferenceConfig, LlamaGenerationResult, LlamaContextInitException;

// Cache
export 'src/cache/kv_cache_manager.dart'
    show LlamaKVCacheManager, LlamaKVCacheSnapshot;

// Sampling
export 'src/sampling/sampling_config.dart'
    show LlamaSamplingConfig, LlamaLogitBias;
```

---

### ✅ Verification Status

```bash
flutter analyze
✅ 0 errors
✅ 0 warnings
ℹ️  Info-level suggestions only (code style)
```

---

### 🛠️ Quick Migration Example

```dart
import 'package:llama_native/llama_native.dart';

void main() async {
  // 1. Backend auto-detects platform
  final backend = LlamaBackend.instance;
  await backend.initialize();
  
  // 2. Load model (sync now)
  final model = LlamaModel.load(
    LlamaModelConfig(modelPath: 'model.gguf')
  );
  
  // 3. Create context (sync now)
  final context = LlamaContext.create(
    model,
    LlamaInferenceConfig.defaultMacOS() // Auto platform detection
  );
  
  // 4. Generate
  await for (final result in context.generateStream(tokens)) {
    print(result.text);
  }
  
  // Cleanup
  context.dispose();
  model.dispose();
  backend.dispose();
}
```

---

### 📝 TODO - Remaining Classes to Rename

- [ ] `ModelConfig` → `LlamaModelConfig`
- [ ] `ModelMetadata` → `LlamaModelMetadata`
- [ ] `ModelLoadException` → `LlamaModelLoadException`
- [ ] `TokenizeException` → `LlamaTokenizeException`
- [ ] `BatchBuilder` → `LlamaBatchBuilder`
- [ ] `LogitBias` → `LlamaLogitBias`

---

### 🎯 Benefits

1. **Consistent Naming** - All public APIs have `Llama` prefix
2. **Better Organization** - One class per file, clear responsibilities
3. **Auto-Detection** - Platform-specific defaults work out of the box
4. **Performance** - Removed compute() overhead for simpler cases
5. **Flexibility** - Easy to customize for specific hardware
6. **Maintainability** - Easier to understand and modify

---

### ⚠️ Notes

- Old class names are completely removed (no deprecation layer)
- Test files updated to use new names
- Example code may need migration
- Documentation files reference old names - to be updated in next pass
