import 'dart:io';
import 'package:llama_native/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/logging/logger.dart';
import 'package:llama_native/src/utils/platform_info.dart';
import 'package:llama_native/src/exceptions/llama_exceptions.dart';
import 'llama_backend_config.dart';

/// 全局硬件加速控制器
///
/// 负责：
/// - 平台检测与后端选择 (Metal/Vulkan/CUDA/CPU)
/// - 初始化/释放 llama 后端
/// - 单例模式实现
class LlamaBackend {
  static LlamaBackend? _instance;
  bool _initialized = false;

  final Logger _logger = Logger('LlamaBackend');
  LlamaBackendConfig _config;

  /// 私有构造函数 (单例模式)
  LlamaBackend._(this._config);

  /// 获取单例实例
  static LlamaBackend get instance {
    if (_instance == null) {
      final config = PlatformInfo.createDefaultBackendConfig();
      _instance = LlamaBackend._(config);
    }
    return _instance!;
  }

  /// 使用自定义配置创建实例
  static LlamaBackend createWithConfig(LlamaBackendConfig config) {
    if (_instance != null) {
      // 如果已存在实例，先释放资源
      _instance!._logger.info('Replacing existing backend with new config');
      if (_instance!._initialized) {
        _instance!.dispose();
      }
    }
    _instance = LlamaBackend._(config);
    _instance!._logger.info('Initialized with custom config');
    return _instance!;
  }

  /// 重置后端实例
  static void reset() {
    if (_instance != null) {
      if (_instance!._initialized) {
        _instance!.dispose();
      }
      _instance = null;
    }
  }

  /// 更新配置（需要重新初始化）
  void updateConfig(LlamaBackendConfig newConfig) {
    if (_initialized) {
      dispose();
    }
    _config = newConfig;
    _logger.info('Config updated, needs re-initialization');
  }

  /// 是否已初始化
  bool get isInitialized => _initialized;

  /// 当前平台（委托给 LlamaPlatformInfo）
  String get currentPlatform => PlatformInfo.currentPlatform;

  /// 检测硬件加速支持（委托给 LlamaPlatformInfo）
  HardwareAcceleration detectHardwareAcceleration() {
    return PlatformInfo.detectHardwareAcceleration();
  }

  /// 初始化后端
  Future<void> initialize() async {
    if (_initialized) {
      _logger.debug('Backend already initialized');
      return;
    }

    _logger.info('Initializing LlamaBackend for $currentPlatform');
    _logger.info('Config: gpu_layers=${_config.gpuLayers}, mmap=${_config.useMmap}');

    try {
      // 检测硬件加速
      final hwAccel = detectHardwareAcceleration();
      _logger.info('Detected hardware acceleration: ${hwAccel.name}');

      // 应用 NUMA 策略
      if (_config.numaStrategy != bindings.ggml_numa_strategy.GGML_NUMA_STRATEGY_DISABLED) {
        try {
          bindings.llama_numa_init(_config.numaStrategy);
          _logger.debug('NUMA strategy applied: ${_config.numaStrategy.name}');
        } catch (e) {
          _logger.warning('Failed to apply NUMA strategy: $e');
          // 继续执行，NUMA 失败不影响后端初始化
        }
      }

      // 初始化 llama 后端
      try {
        bindings.llama_backend_init();
      } catch (e) {
        throw LlamaBackendInitException('Failed to initialize llama backend', platform: currentPlatform);
      }

      _initialized = true;
      _logger.info('✅ Backend initialized successfully');
    } catch (e) {
      _logger.error('Failed to initialize backend: $e');
      if (e is LlamaBackendInitException) {
        rethrow;
      }
      throw LlamaBackendInitException(e.toString(), platform: currentPlatform);
    }
  }

  /// 释放后端资源
  void dispose() {
    if (!_initialized) return;

    _logger.info('Disposing backend...');
    bindings.llama_backend_free();
    _initialized = false;
    _logger.info('✅ Backend disposed');
  }

  /// 获取模型参数 (基于当前配置)
  bindings.llama_model_params getModelParams() {
    return _config.toModelParams();
  }

  /// 获取上下文参数
  bindings.llama_context_params getContextParams({required int nCtx, required int nBatch, required int nThreads}) {
    return _config.toContextParams(nCtx: nCtx, nBatch: nBatch, nThreads: nThreads);
  }
}
