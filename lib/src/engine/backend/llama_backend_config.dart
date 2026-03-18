import 'dart:io';
import 'package:llama_native/src/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/utils/platform_info.dart';

enum FlashAttentionType { auto, disabled, enabled }

class LlamaBackendConfig {
  final int gpuLayers;
  final int mainGpu;
  final bool useMmap;
  final bool useMlock;
  final bindings.ggml_numa_strategy numaStrategy;
  final FlashAttentionType flashAttention;
  final bindings.ggml_type? kvCacheTypeK;
  final bindings.ggml_type? kvCacheTypeV;

  const LlamaBackendConfig({
    this.gpuLayers = 0,
    this.mainGpu = 0,
    this.useMmap = true,
    this.useMlock = false,
    this.numaStrategy = bindings.ggml_numa_strategy.GGML_NUMA_STRATEGY_DISABLED,
    this.flashAttention = FlashAttentionType.auto,
    this.kvCacheTypeK,
    this.kvCacheTypeV,
  });

  factory LlamaBackendConfig.defaultMacOS() {
    final isAppleSilicon =
        Platform.isMacOS &&
        (const String.fromEnvironment('TARGET_ARCH') == 'arm64' || const String.fromEnvironment('TARGET_ARCH').isEmpty);

    final gpuLayers = isAppleSilicon ? PlatformInfo.recommendedGpuLayers() : 0;
    return LlamaBackendConfig(
      gpuLayers: gpuLayers,
      useMmap: true,
      useMlock: false,
      flashAttention: FlashAttentionType.auto,
      kvCacheTypeK: bindings.ggml_type.GGML_TYPE_Q8_0,
      kvCacheTypeV: bindings.ggml_type.GGML_TYPE_Q8_0,
    );
  }

  factory LlamaBackendConfig.defaultAndroid() {
    final gpuLayers = PlatformInfo.recommendedGpuLayers();
    return LlamaBackendConfig(
      gpuLayers: gpuLayers,
      useMmap: true,
      useMlock: false,
      flashAttention: FlashAttentionType.auto,
      kvCacheTypeK: bindings.ggml_type.GGML_TYPE_Q8_0,
      kvCacheTypeV: bindings.ggml_type.GGML_TYPE_Q8_0,
    );
  }

  factory LlamaBackendConfig.defaultWindows() {
    final gpuLayers = PlatformInfo.recommendedGpuLayers();
    return LlamaBackendConfig(
      gpuLayers: gpuLayers,
      useMmap: true,
      useMlock: false,
      flashAttention: FlashAttentionType.auto,
      kvCacheTypeK: bindings.ggml_type.GGML_TYPE_Q8_0,
      kvCacheTypeV: bindings.ggml_type.GGML_TYPE_Q8_0,
    );
  }

  factory LlamaBackendConfig.defaultLinux() {
    final gpuLayers = PlatformInfo.recommendedGpuLayers();
    return LlamaBackendConfig(
      gpuLayers: gpuLayers,
      useMmap: true,
      useMlock: false,
      flashAttention: FlashAttentionType.auto,
      kvCacheTypeK: bindings.ggml_type.GGML_TYPE_Q8_0,
      kvCacheTypeV: bindings.ggml_type.GGML_TYPE_Q8_0,
    );
  }

  factory LlamaBackendConfig.defaultIOS() {
    final gpuLayers = PlatformInfo.recommendedGpuLayers();
    return LlamaBackendConfig(
      gpuLayers: gpuLayers,
      useMmap: true,
      useMlock: false,
      flashAttention: FlashAttentionType.auto,
      kvCacheTypeK: bindings.ggml_type.GGML_TYPE_Q8_0,
      kvCacheTypeV: bindings.ggml_type.GGML_TYPE_Q8_0,
    );
  }

  factory LlamaBackendConfig.forGpuLayers(int gpuLayers) {
    return LlamaBackendConfig(
      gpuLayers: gpuLayers,
      useMmap: true,
      useMlock: false,
      flashAttention: FlashAttentionType.auto,
      kvCacheTypeK: bindings.ggml_type.GGML_TYPE_Q8_0,
      kvCacheTypeV: bindings.ggml_type.GGML_TYPE_Q8_0,
    );
  }

  LlamaBackendConfig copyWith({
    int? gpuLayers,
    int? mainGpu,
    bool? useMmap,
    bool? useMlock,
    bindings.ggml_numa_strategy? numaStrategy,
    FlashAttentionType? flashAttention,
    bindings.ggml_type? kvCacheTypeK,
    bindings.ggml_type? kvCacheTypeV,
  }) {
    return LlamaBackendConfig(
      gpuLayers: gpuLayers ?? this.gpuLayers,
      mainGpu: mainGpu ?? this.mainGpu,
      useMmap: useMmap ?? this.useMmap,
      useMlock: useMlock ?? this.useMlock,
      numaStrategy: numaStrategy ?? this.numaStrategy,
      flashAttention: flashAttention ?? this.flashAttention,
      kvCacheTypeK: kvCacheTypeK ?? this.kvCacheTypeK,
      kvCacheTypeV: kvCacheTypeV ?? this.kvCacheTypeV,
    );
  }

  bindings.llama_model_params toModelParams() {
    final params = bindings.llama_model_default_params();
    params.n_gpu_layers = gpuLayers;
    params.main_gpu = mainGpu;
    params.use_mmap = useMmap;
    params.use_mlock = useMlock;
    return params;
  }

  bindings.llama_context_params toContextParams({
    required int nCtx,
    required int nBatch,
    required int nUBatch,
    required int nThreads,
  }) {
    final params = bindings.llama_context_default_params();
    params.n_ctx = nCtx;
    params.n_batch = nBatch;
    params.n_ubatch = nUBatch;
    params.n_threads = nThreads;
    params.n_threads_batch = nThreads;
    params.no_perf = false;

    switch (flashAttention) {
      case FlashAttentionType.auto:
        params.flash_attn_typeAsInt = bindings.llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_AUTO.value;
        break;
      case FlashAttentionType.disabled:
        params.flash_attn_typeAsInt = bindings.llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_DISABLED.value;
        break;
      case FlashAttentionType.enabled:
        params.flash_attn_typeAsInt = bindings.llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_ENABLED.value;
        break;
    }

    if (kvCacheTypeK != null) {
      params.type_kAsInt = kvCacheTypeK!.value;
    }
    if (kvCacheTypeV != null) {
      params.type_vAsInt = kvCacheTypeV!.value;
    }

    return params;
  }
}
