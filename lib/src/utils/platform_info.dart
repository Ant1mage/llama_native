import 'dart:io';
import 'package:llama_native/src/llama_native_bindings.dart' as bindings;

enum HardwareAcceleration { metal, cuda, vulkan, cpu }

class PlatformInfo {
  static String get currentPlatform {
    if (Platform.isMacOS) return 'macOS';
    if (Platform.isIOS) return 'iOS';
    if (Platform.isAndroid) return 'Android';
    if (Platform.isWindows) return 'Windows';
    if (Platform.isLinux) return 'Linux';
    return 'Unknown';
  }

  static bool get isAppleSilicon {
    if (!Platform.isMacOS) return false;
    return const String.fromEnvironment('TARGET_ARCH') == 'arm64' ||
        const String.fromEnvironment('TARGET_ARCH').isEmpty;
  }

  static bool get supportsGpuOffload {
    return bindings.llama_supports_gpu_offload();
  }

  static int? _cachedSystemMemoryMB;
  static int? _cachedAvailableMemoryMB;

  static int get systemMemoryMB {
    if (_cachedSystemMemoryMB != null) return _cachedSystemMemoryMB!;
    _cachedSystemMemoryMB = _getSystemMemoryMB();
    return _cachedSystemMemoryMB!;
  }

  static int get availableMemoryMB {
    if (_cachedAvailableMemoryMB != null) return _cachedAvailableMemoryMB!;
    _cachedAvailableMemoryMB = _getAvailableMemoryMB();
    return _cachedAvailableMemoryMB!;
  }

  static int _getSystemMemoryMB() {
    try {
      if (Platform.isMacOS) {
        final result = Process.runSync('sysctl', ['-n', 'hw.memsize']);
        if (result.exitCode == 0) {
          final bytes = int.tryParse(result.stdout.toString().trim());
          if (bytes != null) return bytes ~/ (1024 * 1024);
        }
      } else if (Platform.isLinux) {
        final file = File('/proc/meminfo');
        if (file.existsSync()) {
          final content = file.readAsStringSync();
          final match = RegExp(r'MemTotal:\s+(\d+)').firstMatch(content);
          if (match != null) {
            return int.parse(match.group(1)!) ~/ 1024;
          }
        }
      } else if (Platform.isWindows) {
        final result = Process.runSync('wmic', ['OS', 'get', 'TotalVisibleMemorySize', '/Value']);
        if (result.exitCode == 0) {
          final match = RegExp(r'TotalVisibleMemorySize=(\d+)').firstMatch(result.stdout.toString());
          if (match != null) {
            return int.parse(match.group(1)!) ~/ 1024;
          }
        }
      }
    } catch (_) {}

    if (Platform.isMacOS) return isAppleSilicon ? 16 * 1024 : 8 * 1024;
    if (Platform.isIOS) return 4 * 1024;
    if (Platform.isAndroid) return 6 * 1024;
    if (Platform.isWindows || Platform.isLinux) return 16 * 1024;
    return 8 * 1024;
  }

  static int _getAvailableMemoryMB() {
    try {
      if (Platform.isMacOS) {
        final result = Process.runSync('vm_stat', []);
        if (result.exitCode == 0) {
          final output = result.stdout.toString();
          final freeMatch = RegExp(r'free\s*=\s*(\d+)').firstMatch(output);
          final inactiveMatch = RegExp(r'inactive\s*=\s*(\d+)').firstMatch(output);
          if (freeMatch != null && inactiveMatch != null) {
            final free = int.parse(freeMatch.group(1)!);
            final inactive = int.parse(inactiveMatch.group(1)!);
            return ((free + inactive) * 4096) ~/ (1024 * 1024);
          }
        }
      } else if (Platform.isLinux) {
        final file = File('/proc/meminfo');
        if (file.existsSync()) {
          final content = file.readAsStringSync();
          final availableMatch = RegExp(r'MemAvailable:\s+(\d+)').firstMatch(content);
          if (availableMatch != null) {
            return int.parse(availableMatch.group(1)!) ~/ 1024;
          }
        }
      }
    } catch (_) {}

    return (systemMemoryMB * 0.5).toInt();
  }

  static int get availableVRAMMB {
    if (Platform.isMacOS && isAppleSilicon) {
      return (availableMemoryMB * 0.7).toInt();
    }
    if (Platform.isIOS) return (availableMemoryMB * 0.5).toInt();
    if (Platform.isAndroid) return 2048;
    if (Platform.isWindows || Platform.isLinux) {
      return _getCudaVRAMMB() ?? 4096;
    }
    return 0;
  }

  static int? _getCudaVRAMMB() {
    return 8192;
  }

  static int recommendedGpuLayers({int? modelSizeMB, int? modelLayers}) {
    if (!supportsGpuOffload) return 0;

    final vram = availableVRAMMB;
    final mem = availableMemoryMB;

    int estimatedModelSize = modelSizeMB ?? 0;
    if (estimatedModelSize == 0) {
      estimatedModelSize = (mem * 0.3).toInt();
    }

    int usableVram = vram - estimatedModelSize;
    if (usableVram < 512) return 0;

    if (modelLayers != null) {
      final vramPerLayer = (estimatedModelSize * 1.5) / modelLayers;
      final maxLayers = (usableVram / vramPerLayer).floor();
      return maxLayers.clamp(0, modelLayers);
    }

    if (Platform.isMacOS && isAppleSilicon) {
      if (mem >= 32 * 1024) return 99;
      if (mem >= 24 * 1024) return 80;
      if (mem >= 16 * 1024) return 60;
      if (mem >= 12 * 1024) return 40;
      if (mem >= 8 * 1024) return 30;
      if (mem >= 6 * 1024) return 20;
      return 10;
    }

    if (Platform.isIOS) {
      if (mem >= 8 * 1024) return 40;
      if (mem >= 6 * 1024) return 30;
      if (mem >= 4 * 1024) return 20;
      return 10;
    }

    if (usableVram >= 12 * 1024) return 99;
    if (usableVram >= 8 * 1024) return 60;
    if (usableVram >= 6 * 1024) return 40;
    if (usableVram >= 4 * 1024) return 25;
    if (usableVram >= 2 * 1024) return 10;
    return 0;
  }

  static int recommendedContextLength({int? modelSizeMB}) {
    final mem = availableMemoryMB;

    int estimatedModelSize = modelSizeMB ?? 0;
    if (estimatedModelSize == 0) {
      estimatedModelSize = (mem * 0.3).toInt();
    }

    int usableMem = mem - estimatedModelSize;
    if (usableMem < 512) return 512;

    if (Platform.isIOS || Platform.isAndroid) {
      if (usableMem >= 4 * 1024) return 4096;
      if (usableMem >= 2 * 1024) return 2048;
      return 1024;
    }

    if (Platform.isMacOS && isAppleSilicon) {
      if (usableMem >= 12 * 1024) return 8192;
      if (usableMem >= 8 * 1024) return 4096;
      if (usableMem >= 4 * 1024) return 2048;
      return 1024;
    }

    if (usableMem >= 16 * 1024) return 16384;
    if (usableMem >= 12 * 1024) return 8192;
    if (usableMem >= 8 * 1024) return 4096;
    if (usableMem >= 4 * 1024) return 2048;
    return 1024;
  }

  static int recommendedBatchSize({bool useGpu = true}) {
    if (useGpu && supportsGpuOffload) {
      if (Platform.isMacOS && isAppleSilicon) {
        return 512;
      }
      if (Platform.isIOS) return 64;
      if (availableVRAMMB >= 8 * 1024) return 512;
      if (availableVRAMMB >= 4 * 1024) return 256;
      return 128;
    }

    if (Platform.numberOfProcessors >= 8) return 512;
    if (Platform.numberOfProcessors >= 4) return 256;
    return 128;
  }

  static int recommendedThreads() {
    final cores = Platform.numberOfProcessors;
    if (cores <= 2) return cores;
    if (cores <= 4) return cores - 1;
    return (cores * 0.75).ceil().clamp(2, 16);
  }

  static int recommendedUBatchSize() {
    return 128;
  }

  static HardwareAcceleration detectHardwareAcceleration() {
    if (!supportsGpuOffload) return HardwareAcceleration.cpu;
    if (Platform.isMacOS || Platform.isIOS) return HardwareAcceleration.metal;
    if (Platform.isAndroid) return HardwareAcceleration.vulkan;
    return HardwareAcceleration.cuda;
  }

  static String getHardwareInfo() {
    final buffer = StringBuffer();
    buffer.writeln('Platform: $currentPlatform');
    buffer.writeln('CPU Cores: ${Platform.numberOfProcessors}');
    buffer.writeln('System Memory: ${systemMemoryMB}MB');
    buffer.writeln('Available Memory: ${availableMemoryMB}MB');
    buffer.writeln('Available VRAM: ${availableVRAMMB}MB');
    buffer.writeln('GPU Offload: ${supportsGpuOffload ? 'Supported' : 'Not Supported'}');
    buffer.writeln('Hardware Acceleration: ${detectHardwareAcceleration()}');
    if (Platform.isMacOS) buffer.writeln('Apple Silicon: $isAppleSilicon');
    buffer.writeln('--- Recommended Config ---');
    buffer.writeln('GPU Layers: ${recommendedGpuLayers()}');
    buffer.writeln('Context Length: ${recommendedContextLength()}');
    buffer.writeln('Batch Size: ${recommendedBatchSize()}');
    buffer.writeln('Threads: ${recommendedThreads()}');
    return buffer.toString();
  }
}
