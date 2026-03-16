import 'dart:io';
import 'package:llama_native/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/utils/platform_info.dart';

class LlamaBackendConfig {
  final int gpuLayers;
  final int mainGpu;
  final bool useMmap;
  final bool useMlock;
  final bindings.ggml_numa_strategy numaStrategy;

  const LlamaBackendConfig({
    this.gpuLayers = 0,
    this.mainGpu = 0,
    this.useMmap = true,
    this.useMlock = false,
    this.numaStrategy = bindings.ggml_numa_strategy.GGML_NUMA_STRATEGY_DISABLED,
  });

  factory LlamaBackendConfig.defaultMacOS() {
    final isAppleSilicon =
        Platform.isMacOS &&
        (const String.fromEnvironment('TARGET_ARCH') == 'arm64' || const String.fromEnvironment('TARGET_ARCH').isEmpty);

    return LlamaBackendConfig(gpuLayers: isAppleSilicon ? 99 : 0, useMmap: true, useMlock: false);
  }

  factory LlamaBackendConfig.defaultAndroid() {
    final supportsVulkan = PlatformInfo.supportsGpuOffload;
    return LlamaBackendConfig(gpuLayers: supportsVulkan ? 20 : 0, useMmap: true, useMlock: false);
  }

  factory LlamaBackendConfig.defaultWindows() {
    final supportsCuda = PlatformInfo.supportsGpuOffload;
    return LlamaBackendConfig(gpuLayers: supportsCuda ? 35 : 0, useMmap: true, useMlock: false);
  }

  factory LlamaBackendConfig.defaultLinux() {
    final supportsGpu = PlatformInfo.supportsGpuOffload;
    return LlamaBackendConfig(gpuLayers: supportsGpu ? 35 : 0, useMmap: true, useMlock: false);
  }

  factory LlamaBackendConfig.defaultIOS() {
    return LlamaBackendConfig(gpuLayers: PlatformInfo.supportsGpuOffload ? 40 : 0, useMmap: true, useMlock: false);
  }

  factory LlamaBackendConfig.forGpuLayers(int gpuLayers) {
    return LlamaBackendConfig(gpuLayers: gpuLayers, useMmap: true, useMlock: false);
  }

  LlamaBackendConfig copyWith({
    int? gpuLayers,
    int? mainGpu,
    bool? useMmap,
    bool? useMlock,
    bindings.ggml_numa_strategy? numaStrategy,
  }) {
    return LlamaBackendConfig(
      gpuLayers: gpuLayers ?? this.gpuLayers,
      mainGpu: mainGpu ?? this.mainGpu,
      useMmap: useMmap ?? this.useMmap,
      useMlock: useMlock ?? this.useMlock,
      numaStrategy: numaStrategy ?? this.numaStrategy,
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
    return params;
  }
}
