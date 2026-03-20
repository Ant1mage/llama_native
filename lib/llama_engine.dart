import 'dart:async';

import 'package:llama_native/src/llama_isolate.dart';
import 'package:llama_native/src/engine/context/token_generation.dart';
import 'package:llama_native/src/engine/context/performance_metrics.dart';
import 'package:llama_native/src/engine/sampling/sampling_config.dart';
import 'package:llama_native/src/utils/platform_info.dart';
import 'package:llama_native/src/log/logger.dart';

enum LoadState { idle, initializing, loading, ready, error }

enum LoadProgress { initializing, allocatingMemory, loadingModel, creatingContext, ready }

class LlamaEngine {
  final Logger _logger = Logger('LlamaEngine');
  LlamaIsolate? _isolate;
  LoadState _state = LoadState.idle;
  String? _error;
  String? _modelPath;

  final _stateController = StreamController<LoadState>.broadcast();
  final _progressController = StreamController<LoadProgress>.broadcast();

  LoadState get state => _state;
  String? get error => _error;
  String? get modelPath => _modelPath;
  bool get isReady => _state == LoadState.ready;
  bool get isGenerating => _isolate?.isGenerating ?? false;

  Stream<LoadState> get onStateChange => _stateController.stream;
  Stream<LoadProgress> get onProgress => _progressController.stream;

  Future<bool> load(String modelPath, {Map<String, dynamic>? config}) async {
    if (_state == LoadState.loading || _state == LoadState.ready) {
      return _state == LoadState.ready;
    }

    _modelPath = modelPath;
    _setState(LoadState.initializing);
    _setProgress(LoadProgress.initializing);

    try {
      _logger.info('开始加载模型: $modelPath');

      _isolate = LlamaIsolate();
      await _isolate!.initialize();
      _logger.debug('Isolate初始化完成');

      _setProgress(LoadProgress.allocatingMemory);

      final nCtx = config?['contextLength'] as int? ?? PlatformInfo.recommendedContextLength();
      final nGpuLayers = config?['gpuLayers'] as int? ?? PlatformInfo.recommendedGpuLayers();
      final nThreads = config?['threads'] as int? ?? PlatformInfo.recommendedThreads();
      final nBatch = config?['batchSize'] as int? ?? PlatformInfo.recommendedBatchSize();
      final nUBatch = config?['uBatchSize'] as int? ?? PlatformInfo.recommendedUBatchSize();

      _logger.debug('配置: contextLength=$nCtx, gpuLayers=$nGpuLayers, threads=$nThreads');

      final samplingMap = config?['sampling'] as Map<String, dynamic>?;
      final sampling = SamplingConfig(
        temperature: samplingMap?['temperature'] as double? ?? 0.8,
        topK: samplingMap?['topK'] as int? ?? 40,
        topP: samplingMap?['topP'] as double? ?? 0.95,
      );

      _setProgress(LoadProgress.loadingModel);

      final isolateConfig = LlamaIsolateConfig(
        modelPath: modelPath,
        nCtx: nCtx,
        nBatch: nBatch,
        nUBatch: nUBatch,
        nThreads: nThreads,
        nGpuLayers: nGpuLayers,
        sampling: sampling,
      );

      _setProgress(LoadProgress.creatingContext);

      final success = await _isolate!.loadModel(isolateConfig);

      if (success) {
        _setProgress(LoadProgress.ready);
        _setState(LoadState.ready);
        _logger.info('模型加载成功');
        return true;
      } else {
        _error = 'Failed to load model';
        _setState(LoadState.error);
        _logger.error('模型加载失败');
        return false;
      }
    } catch (e) {
      _error = e.toString();
      _setState(LoadState.error);
      _logger.error('模型加载异常: $e');
      return false;
    }
  }

  Future<List<int>> tokenize(String text, {bool addBos = false}) async {
    if (!isReady || _isolate == null) {
      throw StateError('Engine not ready');
    }
    _logger.debug('Tokenize文本: ${text.length}字符');
    return _isolate!.tokenize(text, addBos: addBos);
  }

  Future<String> applyChatTemplate(List<Map<String, String>> messages) async {
    if (!isReady || _isolate == null) {
      throw StateError('Engine not ready');
    }
    _logger.debug('应用Chat模板: ${messages.length}条消息');
    return _isolate!.applyChatTemplate(messages);
  }

  Stream<TokenGeneration> generate(List<int> tokens, {int maxTokens = 1024}) {
    if (!isReady || _isolate == null) {
      throw StateError('Engine not ready');
    }
    _logger.debug('生成Token: ${tokens.length}个, 最大$maxTokens个');
    return _isolate!.generate(tokens, maxTokens: maxTokens);
  }

  void stop() {
    _logger.info('停止生成');
    _isolate?.stop();
  }

  Future<void> stopAsync() async {
    _logger.info('异步停止生成');
    _isolate?.stop();
    await Future.doWhile(() async {
      await Future.delayed(const Duration(milliseconds: 50));
      return _isolate?.isGenerating ?? false;
    });
  }

  Future<void> reset() async {
    if (_isolate != null && _isolate!.isModelLoaded) {
      _logger.info('重置引擎');
      await _isolate!.reset();
    }
  }

  Future<void> setKeepPrefixTokens(List<int> tokens) async {
    if (_isolate != null && _isolate!.isModelLoaded) {
      _logger.debug('设置保留前缀Token: ${tokens.length}个');
      await _isolate!.setKeepPrefixTokens(tokens);
    }
  }

  Future<PerformanceMetrics> getPerformanceMetrics() async {
    if (_isolate == null || !_isolate!.isModelLoaded) {
      return PerformanceMetrics.empty();
    }
    return _isolate!.getPerformanceMetrics();
  }

  Future<List<double>> embed(String text) async {
    if (!isReady || _isolate == null) {
      throw StateError('Engine not ready');
    }
    _logger.debug('生成嵌入向量: ${text.length}字符');
    return _isolate!.embed(text);
  }

  Future<void> dispose() async {
    _logger.info('释放引擎资源');
    await _isolate?.dispose();
    _isolate = null;
    _setState(LoadState.idle);
  }

  void _setState(LoadState state) {
    _state = state;
    _stateController.add(state);
  }

  void _setProgress(LoadProgress progress) {
    _progressController.add(progress);
  }
}
