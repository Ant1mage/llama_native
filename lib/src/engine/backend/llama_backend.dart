import 'dart:io';
import 'package:llama_native/src/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/log/logger.dart';
import 'package:llama_native/src/utils/platform_info.dart';
import 'package:llama_native/src/engine/exceptions/llama_exceptions.dart';
import 'llama_backend_config.dart';

class LlamaBackend {
  static LlamaBackend? _instance;

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
      _instance!._logger.info('使用默认配置初始化后端');
    }
    return _instance!;
  }

  static void configure(LlamaBackendConfig config) {
    _instance = LlamaBackend._(config);
    _instance!._logger.info('使用自定义配置创建后端');
  }

  static void reset() {
    _instance = null;
  }

  void initialize() {
    _logger.info('初始化后端');
    try {
      bindings.llama_backend_init();

      final gpuLayers = _config.gpuLayers;
      if (gpuLayers > 0) {
        _logger.info('使用GPU加速，层数: $gpuLayers');
      } else if (gpuLayers == -1) {
        _logger.info('使用GPU加速，卸载所有层到GPU');
      } else {
        _logger.info('使用CPU模式');
      }

      _logger.info('后端初始化完成');
    } catch (e) {
      throw LlamaException.backend(e.toString(), platform: PlatformInfo.currentPlatform);
    }
  }

  bindings.llama_model_params getModelParams() {
    final params = bindings.llama_model_default_params();

    params.n_gpu_layers = _config.gpuLayers;
    params.split_modeAsInt = bindings.llama_split_mode.LLAMA_SPLIT_MODE_LAYER.value;

    return params;
  }

  bindings.llama_context_params getContextParams({
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
    params.n_seq_max = 1;
    params.embeddings = false;

    return params;
  }

  void dispose() {
    _logger.info('释放后端资源');
    bindings.llama_backend_free();
    _logger.info('后端已释放');
  }
}
