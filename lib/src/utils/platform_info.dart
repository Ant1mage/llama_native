import 'dart:io';
import 'package:llama_native/llama_native.dart';
import 'package:llama_native/llama_native_bindings.dart' as bindings;


/// 硬件加速类型枚举
enum HardwareAcceleration {
  /// Apple Metal (macOS/iOS)
  metal,

  /// NVIDIA CUDA (Windows/Linux)
  cuda,

  /// Vulkan (Android/Linux)
  vulkan,

  /// 纯 CPU (通用回退)
  cpu,
}

/// Llama 平台信息工具类
class PlatformInfo {
  /// 当前平台
  static String get currentPlatform {
    if (Platform.isMacOS) return 'macOS';
    if (Platform.isIOS) return 'iOS';
    if (Platform.isAndroid) return 'Android';
    if (Platform.isWindows) return 'Windows';
    if (Platform.isLinux) return 'Linux';
    return 'Unknown';
  }

  /// 是否为 Apple Silicon (M1/M2/M3)
  static bool get isAppleSilicon {
    if (!Platform.isMacOS) return false;
    // 通过 uname 检测架构
    return const String.fromEnvironment('TARGET_ARCH') == 'arm64' ||
        const String.fromEnvironment('TARGET_ARCH').isEmpty;
  }

  /// 是否支持 GPU 卸载
  static bool get supportsGpuOffload {
    return bindings.llama_supports_gpu_offload();
  }

  /// 获取系统内存总量（MB）- 估算值
  static int _getSystemMemoryMB() {
    // Dart 无法直接获取系统内存，使用平台默认值
    if (Platform.isMacOS) {
      if (isAppleSilicon) {
        // M 系列芯片通常 8GB-192GB
        return 16 * 1024; // 假设 16GB
      }
      return 8 * 1024; // Intel Mac 假设 8GB
    } else if (Platform.isIOS) {
      return 4 * 1024; // iOS 设备假设 4GB
    } else if (Platform.isAndroid) {
      return 6 * 1024; // Android 设备假设 6GB
    } else if (Platform.isWindows || Platform.isLinux) {
      return 16 * 1024; // Windows/Linux 假设 16GB
    }
    return 8 * 1024; // 默认 8GB
  }

  /// 获取可用显存估算（MB）
  static int _getAvailableVRAM() {
    if (Platform.isMacOS && isAppleSilicon) {
      // Apple Silicon 统一内存，通常可使用 50-75%
      final systemMem = _getSystemMemoryMB();
      return (systemMem * 0.6).toInt(); // 60% 可用于 GPU
    } else if (Platform.isAndroid) {
      // Android Vulkan 通常 2-4GB
      return 2 * 1024;
    } else if (Platform.isWindows || Platform.isLinux) {
      // Windows/Linux CUDA 通常 4-24GB
      return 8 * 1024; // 假设 8GB
    }
    return 0; // CPU only
  }

  /// 推荐的最大 GPU 层数
  static int recommendedGpuLayers() {
    if (isAppleSilicon) return 100; // M 系列芯片可全量卸载
    if (Platform.isIOS) return 50; // iOS 保守设置

    // 其他平台根据显存计算
    if (supportsGpuOffload) {
      final vram = _getAvailableVRAM();
      // 每层大约需要 100-200MB，保守估计
      if (vram >= 8 * 1024) return 100; // 8GB+ 全量
      if (vram >= 4 * 1024) return 50; // 4GB 半量
      if (vram >= 2 * 1024) return 25; // 2GB 部分
    }

    return 0; // 不支持 GPU 或显存不足
  }

  /// 推荐的上下文长度
  static int recommendedContextLength() {
    if (isAppleSilicon) return 8192; // M 系列支持大上下文
    if (Platform.isIOS) return 2048; // iOS 限制上下文

    final vram = _getAvailableVRAM();
    if (vram >= 8 * 1024) return 8192;
    if (vram >= 4 * 1024) return 4096;
    return 2048;
  }

  /// 推荐的批次大小
  static int recommendedBatchSize() {
    if (isAppleSilicon) return 1024;
    if (Platform.isIOS) return 256;

    final vram = _getAvailableVRAM();
    if (vram >= 8 * 1024) return 1024;
    if (vram >= 4 * 1024) return 512;
    return 256;
  }

  /// 推荐的线程数
  static int recommendedThreads() {
    if (Platform.isMacOS || Platform.isLinux) {
      // Unix-like 系统可以获取 CPU 核心数
      try {
        // 使用物理核心数的 75%
        final cores = Platform.numberOfProcessors;
        return (cores * 0.75).ceil().clamp(2, 16);
      } catch (_) {
        return 4;
      }
    } else if (Platform.isWindows) {
      try {
        final cores = Platform.numberOfProcessors;
        return (cores * 0.75).ceil().clamp(2, 16);
      } catch (_) {
        return 4;
      }
    } else if (Platform.isAndroid) {
      // Android 通常 4-8 核心
      return Platform.numberOfProcessors.clamp(2, 4);
    } else if (Platform.isIOS) {
      // iOS 通常 4-6 核心
      return Platform.numberOfProcessors.clamp(2, 4);
    }
    return 4;
  }

  /// 创建默认后端配置
  static LlamaBackendConfig createDefaultBackendConfig() {
    if (Platform.isMacOS) {
      return LlamaBackendConfig.defaultMacOS();
    } else if (Platform.isAndroid) {
      return LlamaBackendConfig.defaultAndroid();
    } else if (Platform.isWindows) {
      return LlamaBackendConfig.defaultWindows();
    } else if (Platform.isLinux) {
      return LlamaBackendConfig.defaultLinux();
    } else if (Platform.isIOS) {
      return LlamaBackendConfig.defaultIOS();
    }
    return const LlamaBackendConfig();
  }

  /// 检测硬件加速支持
  static HardwareAcceleration detectHardwareAcceleration() {
    if (!supportsGpuOffload) {
      return HardwareAcceleration.cpu;
    }

    if (Platform.isMacOS || Platform.isIOS) {
      return HardwareAcceleration.metal;
    } else if (Platform.isAndroid) {
      return HardwareAcceleration.vulkan;
    } else if (Platform.isWindows) {
      // Windows 优先检测 CUDA
      try {
        // 这里可以添加 CUDA 检测逻辑
        return HardwareAcceleration.cuda;
      } catch (_) {
        // 回退到其他 GPU 支持
        return HardwareAcceleration.vulkan;
      }
    } else if (Platform.isLinux) {
      // Linux 检查 CUDA 或 Vulkan
      try {
        // 这里可以添加 CUDA 检测逻辑
        return HardwareAcceleration.cuda;
      } catch (_) {
        return HardwareAcceleration.vulkan;
      }
    }
    return HardwareAcceleration.cpu;
  }
}
