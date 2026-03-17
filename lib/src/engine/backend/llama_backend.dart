import 'dart:io';
import 'package:llama_native/src/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/log/logger.dart';
import 'package:llama_native/src/utils/platform_info.dart';
import 'package:llama_native/src/engine/exceptions/llama_exceptions.dart';
import 'llama_backend_config.dart';

class LlamaBackend {
  static LlamaBackend? _instance;
  bool _initialized = false;

  final Logger _logger = Logger('LlamaBackend');
  LlamaBackendConfig _config;

  LlamaBackend._(this._config);

  static LlamaBackend get instance {
    if (_instance == null) {
      LlamaBackendConfig config;
      if (Platform.isMacOS) {
        config = LlamaBackendConfig.defaultMacOS();
      } else if (Platform.isAndroid) {
        config = LlamaBackendConfig.defaultAndroid();
      } else if (Platform.isWindows) {
        config = LlamaBackendConfig.defaultWindows();
      } else if (Platform.isLinux) {
        config = LlamaBackendConfig.defaultLinux();
      } else if (Platform.isIOS) {
        config = LlamaBackendConfig.defaultIOS();
      } else {
        config = const LlamaBackendConfig();
      }
      _instance = LlamaBackend._(config);
    }
    return _instance!;
  }

  static LlamaBackend createWithConfig(LlamaBackendConfig config) {
    if (_instance != null) {
      _instance!._logger.info('Replacing existing backend with new config');
      if (_instance!._initialized) {
        _instance!.dispose();
      }
    }
    _instance = LlamaBackend._(config);
    _instance!._logger.info('Initialized with custom config');
    return _instance!;
  }

  static void reset() {
    if (_instance != null) {
      if (_instance!._initialized) {
        _instance!.dispose();
      }
      _instance = null;
    }
  }

  void updateConfig(LlamaBackendConfig newConfig) {
    if (_initialized) {
      dispose();
    }
    _config = newConfig;
    _logger.info('Config updated, needs re-initialization');
  }

  bool get isInitialized => _initialized;

  String get currentPlatform => PlatformInfo.currentPlatform;

  HardwareAcceleration detectHardwareAcceleration() {
    return PlatformInfo.detectHardwareAcceleration();
  }

  Future<void> initialize() async {
    if (_initialized) {
      _logger.debug('Backend already initialized');
      return;
    }

    _logger.info('Initializing LlamaBackend for $currentPlatform');
    _logger.info('\n${PlatformInfo.getHardwareInfo()}');
    _logger.info('Config: gpu_layers=${_config.gpuLayers}, mmap=${_config.useMmap}');

    try {
      final hwAccel = detectHardwareAcceleration();
      _logger.info('Detected hardware acceleration: ${hwAccel.name}');

      if (_config.numaStrategy != bindings.ggml_numa_strategy.GGML_NUMA_STRATEGY_DISABLED) {
        try {
          bindings.llama_numa_init(_config.numaStrategy);
          _logger.debug('NUMA strategy applied: ${_config.numaStrategy.name}');
        } catch (e) {
          _logger.warning('Failed to apply NUMA strategy: $e');
        }
      }

      try {
        bindings.llama_backend_init();
      } catch (e) {
        throw LlamaBackendInitException('Failed to initialize llama backend', platform: currentPlatform);
      }

      _initialized = true;
      _logger.info('Backend initialized successfully');
    } catch (e) {
      _logger.error('Failed to initialize backend: $e');
      if (e is LlamaBackendInitException) {
        rethrow;
      }
      throw LlamaBackendInitException(e.toString(), platform: currentPlatform);
    }
  }

  void dispose() {
    if (!_initialized) return;

    _logger.info('Disposing backend...');
    bindings.llama_backend_free();
    _initialized = false;
    _logger.info('Backend disposed');
  }

  bindings.llama_model_params getModelParams() {
    return _config.toModelParams();
  }

  bindings.llama_context_params getContextParams({
    required int nCtx,
    required int nBatch,
    required int nUBatch,
    required int nThreads,
  }) {
    return _config.toContextParams(nCtx: nCtx, nBatch: nBatch, nUBatch: nUBatch, nThreads: nThreads);
  }
}
