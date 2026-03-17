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
    if (Platform.isIOS) return 6 * 1024;
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
    final hwAccel = detectHardwareAcceleration();

    if (hwAccel == HardwareAcceleration.cpu) return 0;

    final mem = availableMemoryMB;
    final vram = availableVRAMMB;

    double coeff;
    switch (hwAccel) {
      case HardwareAcceleration.metal:
        coeff = 0.6;
        break;
      case HardwareAcceleration.vulkan:
        coeff = 0.6;
        break;
      case HardwareAcceleration.cuda:
        coeff = 0.85;
        break;
      case HardwareAcceleration.cpu:
        return 0;
    }

    int estimatedModelSize = modelSizeMB ?? (mem * 0.3).toInt();
    final kvCacheMB = (recommendedContextLength() * 0.5).toInt();
    int usableVram = ((vram * coeff - kvCacheMB)).toInt();
    if (usableVram < 512) return 0;

    if (modelLayers != null && modelLayers > 0) {
      final memoryPerLayer = estimatedModelSize / modelLayers;
      final layers = (usableVram / memoryPerLayer).floor();
      return layers.clamp(0, modelLayers);
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

    const int bytesPerToken = 2;
    final int rawContext = (bytesPerToken * mem * 0.2).toInt();

    return rawContext.clamp(1024, 8192);
  }

  static int recommendedThreads() {
    final cores = Platform.numberOfProcessors;
    return (cores - 2).clamp(1, 6);
  }

  static int recommendedBatchSize({bool useGpu = true}) {
    final mem = systemMemoryMB;

    if (mem > 8 * 1024) return 512;
    return 256;
  }

  static int recommendedUBatchSize() {
    return recommendedBatchSize().clamp(64, 128);
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
