import 'dart:io';
import 'package:llama_native/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/utils/platform_info.dart';

/// Llama 后端配置参数
class LlamaBackendConfig {
  /// GPU 加载层数 (0 = 纯 CPU, 100 = 全量 GPU)
  final int gpuLayers;

  /// 使用内存映射 (mmap) 优化
  final bool useMmap;

  /// 锁定内存防止交换 (mlock)
  final bool useMlock;

  /// NUMA 策略
  final bindings.ggml_numa_strategy numaStrategy;

  const LlamaBackendConfig({
    this.gpuLayers = 0,
    this.useMmap = true,
    this.useMlock = false,
    this.numaStrategy = bindings.ggml_numa_strategy.GGML_NUMA_STRATEGY_DISABLED,
  });

  /// 创建默认配置 (macOS Metal 优化)
  factory LlamaBackendConfig.defaultMacOS() {
    // 检测 M1/M2 芯片
    final isAppleSilicon =
        Platform.isMacOS &&
        (const String.fromEnvironment('TARGET_ARCH') == 'arm64' || const String.fromEnvironment('TARGET_ARCH').isEmpty);

    return LlamaBackendConfig(gpuLayers: isAppleSilicon ? 100 : 0, useMmap: true, useMlock: false);
  }

  /// Android 默认配置 (Vulkan)
  factory LlamaBackendConfig.defaultAndroid() {
    final supportsVulkan = PlatformInfo.supportsGpuOffload;
    final vram = _estimateAndroidVRAM();

    int gpuLayers = 0;
    if (supportsVulkan) {
      if (vram >= 8 * 1024) {
        gpuLayers = 100;
      } else if (vram >= 4 * 1024) {
        gpuLayers = 50;
      } else if (vram >= 2 * 1024) {
        gpuLayers = 25;
      }
    }

    return LlamaBackendConfig(gpuLayers: gpuLayers, useMmap: true, useMlock: false);
  }

  /// Windows 默认配置 (CUDA)
  factory LlamaBackendConfig.defaultWindows() {
    final supportsCuda = PlatformInfo.supportsGpuOffload;
    final vram = _estimateWindowsVRAM();

    int gpuLayers = 0;
    if (supportsCuda) {
      if (vram >= 12 * 1024) {
        gpuLayers = 100;
      } else if (vram >= 8 * 1024) {
        gpuLayers = 75;
      } else if (vram >= 4 * 1024) {
        gpuLayers = 50;
      }
    }

    return LlamaBackendConfig(gpuLayers: gpuLayers, useMmap: true, useMlock: false);
  }

  /// Linux 默认配置 (CUDA/Vulkan)
  factory LlamaBackendConfig.defaultLinux() {
    final supportsGpu = PlatformInfo.supportsGpuOffload;
    final vram = _estimateLinuxVRAM();

    int gpuLayers = 0;
    if (supportsGpu) {
      if (vram >= 12 * 1024) {
        gpuLayers = 100;
      } else if (vram >= 8 * 1024) {
        gpuLayers = 75;
      } else if (vram >= 4 * 1024) {
        gpuLayers = 50;
      }
    }

    return LlamaBackendConfig(gpuLayers: gpuLayers, useMmap: true, useMlock: false);
  }

  /// iOS 默认配置 (Metal)
  factory LlamaBackendConfig.defaultIOS() {
    // iOS 设备统一使用 Metal，保守设置
    return LlamaBackendConfig(gpuLayers: 50, useMmap: true, useMlock: false);
  }

  /// 估算 Android 显存
  static int _estimateAndroidVRAM() {
    // Android 设备通常 2-8GB 显存
    // 这里使用保守估计
    return 4 * 1024; // 假设 4GB
  }

  /// 估算 Windows 显存
  static int _estimateWindowsVRAM() {
    // Windows 游戏显卡通常 4-24GB
    return 8 * 1024; // 假设 8GB
  }

  /// 估算 Linux 显存
  static int _estimateLinuxVRAM() {
    // Linux 服务器/工作站通常 8-48GB
    return 12 * 1024; // 假设 12GB
  }

  /// 转换为 native model params
  bindings.llama_model_params toModelParams() {
    final params = bindings.llama_model_default_params();
    params.n_gpu_layers = gpuLayers;
    params.use_mmap = useMmap;
    params.use_mlock = useMlock;
    return params;
  }

  /// 转换为 native context params
  bindings.llama_context_params toContextParams({required int nCtx, required int nBatch, required int nThreads}) {
    final params = bindings.llama_context_default_params();
    params.n_ctx = nCtx;
    params.n_batch = nBatch;
    params.n_threads = nThreads;
    params.n_threads_batch = nThreads;
    return params;
  }
}
