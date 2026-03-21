import 'dart:async';
import 'dart:isolate';

import 'package:llama_native/llama_chat_message.dart';
import 'package:llama_native/src/engine/context/inference_config.dart';
import 'package:llama_native/src/engine/context/llama_context.dart';
import 'package:llama_native/src/engine/context/performance_metrics.dart';
import 'package:llama_native/src/engine/context/token_generation.dart';
import 'package:llama_native/src/engine/model/llama_model.dart';
import 'package:llama_native/src/engine/sampling/sampling_config.dart';
import 'package:llama_native/src/engine/tokenizer/llama_tokenizer.dart';
import 'package:llama_native/src/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/log/logger.dart';

enum _MessageType {
  loadModel,
  loadModelResult,
  tokenize,
  tokenizeResult,
  applyChatTemplate,
  applyChatTemplateResult,
  generate,
  tokenGeneration,
  stop,
  stopped,
  error,
  done,
  reset,
  resetResult,
  dispose,
  disposeResult,
  setKeepPrefixTokens,
  setKeepPrefixTokensResult,
  getPerformanceMetrics,
  performanceMetricsResult,
  embed,
  embedResult,
  pauseKVCache,
  pauseKVCacheResult,
  resumeKVCache,
  resumeKVCacheResult,
  prepareForSnapshot,
  prepareForSnapshotResult,
  injectContextTokens,
  injectContextTokensResult,
}

class _IsolateMessage {
  final _MessageType type;
  final dynamic data;

  const _IsolateMessage(this.type, this.data);
}

class LlamaIsolateConfig {
  final String modelPath;
  final int nCtx;
  final int nBatch;
  final int nUBatch;
  final int nThreads;
  final int nGpuLayers;
  final SamplingConfig sampling;

  const LlamaIsolateConfig({
    required this.modelPath,
    this.nCtx = 4096,
    this.nBatch = 512,
    this.nUBatch = 128,
    this.nThreads = 4,
    this.nGpuLayers = 0,
    this.sampling = const SamplingConfig(),
  });
}

class LlamaIsolate {
  Isolate? _isolate;
  SendPort? _sendPort;
  ReceivePort? _receivePort;
  StreamSubscription? _subscription;
  final Logger _logger = Logger('LlamaIsolate');
  bool _isModelLoaded = false;
  bool _isGenerating = false;

  bool get isInitialized => _isolate != null && _sendPort != null;
  bool get isModelLoaded => _isModelLoaded;
  bool get isGenerating => _isGenerating;

  Future<void> initialize() async {
    if (isInitialized) return;

    _receivePort = ReceivePort();
    _isolate = await Isolate.spawn(_isolateEntryPoint, _receivePort!.sendPort, debugName: 'LlamaIsolate');

    final completer = Completer<void>();
    _subscription = _receivePort!.listen((message) {
      if (message is SendPort) {
        _sendPort = message;
        completer.complete();
      } else if (message is _IsolateMessage) {
        _handleMessage(message);
      }
    });

    await completer.future;
    _logger.info('LlamaIsolate初始化完成');
  }

  void _handleMessage(_IsolateMessage message) {
    switch (message.type) {
      case _MessageType.tokenGeneration:
      case _MessageType.stopped:
      case _MessageType.error:
      case _MessageType.done:
        break;
      default:
        break;
    }
  }

  Future<bool> loadModel(LlamaIsolateConfig config) async {
    if (!isInitialized || _sendPort == null) {
      throw StateError('Isolate未初始化');
    }

    final completer = Completer<bool>();
    final responsePort = ReceivePort();

    _sendPort!.send(_IsolateMessage(_MessageType.loadModel, {'config': config, 'responsePort': responsePort.sendPort}));

    responsePort.listen((message) {
      if (message is _IsolateMessage) {
        if (message.type == _MessageType.loadModelResult) {
          _isModelLoaded = message.data['success'] as bool;
          completer.complete(_isModelLoaded);
          responsePort.close();
        } else if (message.type == _MessageType.error) {
          completer.completeError(message.data);
          responsePort.close();
        }
      }
    });

    return completer.future;
  }

  Future<List<int>> tokenize(String text, {bool addBos = false}) async {
    if (!_isModelLoaded || _sendPort == null) {
      throw StateError('模型未加载');
    }

    final completer = Completer<List<int>>();
    final responsePort = ReceivePort();

    _sendPort!.send(
      _IsolateMessage(_MessageType.tokenize, {'text': text, 'addBos': addBos, 'responsePort': responsePort.sendPort}),
    );

    responsePort.listen((message) {
      if (message is _IsolateMessage) {
        if (message.type == _MessageType.tokenizeResult) {
          completer.complete(message.data as List<int>);
          responsePort.close();
        } else if (message.type == _MessageType.error) {
          completer.completeError(message.data);
          responsePort.close();
        }
      }
    });

    return completer.future;
  }

  Future<String> applyChatTemplate(List<Map<String, String>> messages) async {
    if (!_isModelLoaded || _sendPort == null) {
      throw StateError('模型未加载');
    }

    final completer = Completer<String>();
    final responsePort = ReceivePort();

    _sendPort!.send(
      _IsolateMessage(_MessageType.applyChatTemplate, {'messages': messages, 'responsePort': responsePort.sendPort}),
    );

    responsePort.listen((message) {
      if (message is _IsolateMessage) {
        if (message.type == _MessageType.applyChatTemplateResult) {
          completer.complete(message.data as String);
          responsePort.close();
        } else if (message.type == _MessageType.error) {
          completer.completeError(message.data);
          responsePort.close();
        }
      }
    });

    return completer.future;
  }

  Stream<TokenGeneration> generate(List<int> tokens, {int maxTokens = 1024}) async* {
    if (!_isModelLoaded || _sendPort == null) {
      throw StateError('模型未加载');
    }

    _isGenerating = true;
    final responsePort = ReceivePort();
    final controller = StreamController<TokenGeneration>();

    _sendPort!.send(
      _IsolateMessage(_MessageType.generate, {
        'tokens': tokens,
        'maxTokens': maxTokens,
        'responsePort': responsePort.sendPort,
      }),
    );

    responsePort.listen((message) {
      if (message is _IsolateMessage) {
        switch (message.type) {
          case _MessageType.tokenGeneration:
            controller.add(message.data as TokenGeneration);
            break;
          case _MessageType.stopped:
          case _MessageType.done:
            _isGenerating = false;
            controller.close();
            responsePort.close();
            break;
          case _MessageType.error:
            _isGenerating = false;
            controller.addError(message.data);
            controller.close();
            responsePort.close();
            break;
          default:
            break;
        }
      }
    });

    yield* controller.stream;
  }

  void stop() {
    if (_isGenerating && _sendPort != null) {
      _sendPort!.send(_IsolateMessage(_MessageType.stop, null));
    }
  }

  Future<void> reset() async {
    if (!_isModelLoaded || _sendPort == null) return;

    final completer = Completer<void>();
    final responsePort = ReceivePort();

    _sendPort!.send(_IsolateMessage(_MessageType.reset, {'responsePort': responsePort.sendPort}));

    responsePort.listen((message) {
      if (message is _IsolateMessage && message.type == _MessageType.resetResult) {
        completer.complete();
        responsePort.close();
      }
    });

    return completer.future;
  }

  Future<void> setKeepPrefixTokens(List<int> tokens) async {
    if (!_isModelLoaded || _sendPort == null) return;

    final completer = Completer<void>();
    final responsePort = ReceivePort();

    _sendPort!.send(
      _IsolateMessage(_MessageType.setKeepPrefixTokens, {'tokens': tokens, 'responsePort': responsePort.sendPort}),
    );

    responsePort.listen((message) {
      if (message is _IsolateMessage && message.type == _MessageType.setKeepPrefixTokensResult) {
        completer.complete();
        responsePort.close();
      }
    });

    return completer.future;
  }

  Future<PerformanceMetrics> getPerformanceMetrics() async {
    if (!_isModelLoaded || _sendPort == null) {
      return PerformanceMetrics.empty();
    }

    final completer = Completer<PerformanceMetrics>();
    final responsePort = ReceivePort();

    _sendPort!.send(_IsolateMessage(_MessageType.getPerformanceMetrics, {'responsePort': responsePort.sendPort}));

    responsePort.listen((message) {
      if (message is _IsolateMessage && message.type == _MessageType.performanceMetricsResult) {
        final data = message.data as Map<String, dynamic>;
        completer.complete(
          PerformanceMetrics(
            tStartMs: data['tStartMs'] as double,
            tLoadMs: data['tLoadMs'] as double,
            tPromptEvalMs: data['tPromptEvalMs'] as double,
            tEvalMs: data['tEvalMs'] as double,
            nPromptEval: data['nPromptEval'] as int,
            nEval: data['nEval'] as int,
            nReused: data['nReused'] as int,
            tSampleMs: data['tSampleMs'] as double,
            nSample: data['nSample'] as int,
          ),
        );
        responsePort.close();
      }
    });

    return completer.future;
  }

  Future<List<double>> embed(String text) async {
    if (!_isModelLoaded || _sendPort == null) {
      throw StateError('模型未加载');
    }

    final completer = Completer<List<double>>();
    final responsePort = ReceivePort();

    _sendPort!.send(_IsolateMessage(_MessageType.embed, {'text': text, 'responsePort': responsePort.sendPort}));

    responsePort.listen((message) {
      if (message is _IsolateMessage) {
        if (message.type == _MessageType.embedResult) {
          completer.complete(List<double>.from(message.data as List));
          responsePort.close();
        } else if (message.type == _MessageType.error) {
          completer.completeError(message.data);
          responsePort.close();
        }
      }
    });

    return completer.future;
  }

  Future<void> pauseKVCache() async {
    if (!_isModelLoaded || _sendPort == null) return;

    final completer = Completer<void>();
    final responsePort = ReceivePort();

    _sendPort!.send(_IsolateMessage(_MessageType.pauseKVCache, {'responsePort': responsePort.sendPort}));

    responsePort.listen((message) {
      if (message is _IsolateMessage && message.type == _MessageType.pauseKVCacheResult) {
        completer.complete();
        responsePort.close();
      }
    });

    return completer.future;
  }

  Future<void> resumeKVCache() async {
    if (!_isModelLoaded || _sendPort == null) return;

    final completer = Completer<void>();
    final responsePort = ReceivePort();

    _sendPort!.send(_IsolateMessage(_MessageType.resumeKVCache, {'responsePort': responsePort.sendPort}));

    responsePort.listen((message) {
      if (message is _IsolateMessage && message.type == _MessageType.resumeKVCacheResult) {
        completer.complete();
        responsePort.close();
      }
    });

    return completer.future;
  }

  Future<void> prepareForSnapshot() async {
    if (!_isModelLoaded || _sendPort == null) return;

    final completer = Completer<void>();
    final responsePort = ReceivePort();

    _sendPort!.send(_IsolateMessage(_MessageType.prepareForSnapshot, {'responsePort': responsePort.sendPort}));

    responsePort.listen((message) {
      if (message is _IsolateMessage && message.type == _MessageType.prepareForSnapshotResult) {
        completer.complete();
        responsePort.close();
      }
    });

    return completer.future;
  }

  Future<void> injectContextTokens(List<int> tokens) async {
    if (!_isModelLoaded || _sendPort == null) return;

    final completer = Completer<void>();
    final responsePort = ReceivePort();

    _sendPort!.send(
      _IsolateMessage(_MessageType.injectContextTokens, {'tokens': tokens, 'responsePort': responsePort.sendPort}),
    );

    responsePort.listen((message) {
      if (message is _IsolateMessage && message.type == _MessageType.injectContextTokensResult) {
        completer.complete();
        responsePort.close();
      }
    });

    return completer.future;
  }

  Future<void> dispose() async {
    if (_sendPort != null) {
      final completer = Completer<void>();
      final responsePort = ReceivePort();

      _sendPort!.send(_IsolateMessage(_MessageType.dispose, {'responsePort': responsePort.sendPort}));

      responsePort.listen((message) {
        if (message is _IsolateMessage && message.type == _MessageType.disposeResult) {
          completer.complete();
          responsePort.close();
        }
      });

      await completer.future;
    }

    _subscription?.cancel();
    _receivePort?.close();
    _isolate?.kill(priority: Isolate.immediate);

    _isolate = null;
    _sendPort = null;
    _receivePort = null;
    _subscription = null;
    _isModelLoaded = false;
    _isGenerating = false;

    _logger.info('LlamaIsolate已释放');
  }

  static void _isolateEntryPoint(SendPort mainSendPort) {
    final receivePort = ReceivePort();
    mainSendPort.send(receivePort.sendPort);

    final logger = Logger('LlamaIsolateWorker');

    LlamaModel? model;
    LlamaContext? context;
    LlamaTokenizer? chatTokenizer;
    bool shouldStop = false;

    receivePort.listen((message) {
      if (message is _IsolateMessage) {
        switch (message.type) {
          case _MessageType.loadModel:
            _handleLoadModel(message.data, (m, ctx, tokenizer) {
              model = m;
              context = ctx;
              chatTokenizer = tokenizer;
            }, logger);
            break;

          case _MessageType.generate:
            if (context != null) {
              shouldStop = false;
              _handleGenerateAsync(message.data, context!, () => shouldStop, logger);
            }
            break;

          case _MessageType.stop:
            shouldStop = true;
            logger.debug('收到停止信号');
            break;

          case _MessageType.tokenize:
            if (model != null) {
              _handleTokenize(message.data, model!, logger);
            }
            break;

          case _MessageType.applyChatTemplate:
            if (chatTokenizer != null) {
              _handleApplyChatTemplate(message.data, chatTokenizer!, logger);
            }
            break;

          case _MessageType.reset:
            _handleReset(message.data, context, logger);
            break;

          case _MessageType.dispose:
            _handleDispose(message.data, model, context, () {
              model = null;
              context = null;
              chatTokenizer = null;
            }, logger);
            break;

          case _MessageType.setKeepPrefixTokens:
            _handleSetKeepPrefixTokens(message.data, context, logger);
            break;

          case _MessageType.getPerformanceMetrics:
            _handleGetPerformanceMetrics(message.data, context, logger);
            break;

          case _MessageType.embed:
            _handleEmbed(message.data, model, logger);
            break;

          case _MessageType.pauseKVCache:
            _handlePauseKVCache(message.data, context, logger);
            break;

          case _MessageType.resumeKVCache:
            _handleResumeKVCache(message.data, context, logger);
            break;

          case _MessageType.prepareForSnapshot:
            _handlePrepareForSnapshot(message.data, context, logger);
            break;

          case _MessageType.injectContextTokens:
            _handleInjectContextTokens(message.data, context, logger);
            break;

          default:
            break;
        }
      }
    });
  }

  static void _handleLoadModel(
    Map<String, dynamic> data,
    Function(LlamaModel, LlamaContext, LlamaTokenizer) onSuccess,
    Logger logger,
  ) {
    final config = data['config'] as LlamaIsolateConfig;
    final responsePort = data['responsePort'] as SendPort;

    try {
      logger.info('加载模型: ${config.modelPath}');

      // 初始化backend（每个isolate都需要调用一次）
      bindings.llama_backend_init();
      logger.info('Backend初始化完成');

      // 创建模型配置
      final modelConfig = LlamaModelConfig(modelPath: config.modelPath, gpuLayers: config.nGpuLayers);

      final model = LlamaModel.load(modelConfig);
      logger.info('模型加载完成');

      final inferenceConfig = InferenceConfig(
        nCtx: config.nCtx,
        nBatch: config.nBatch,
        nUBatch: config.nUBatch,
        nThreads: config.nThreads,
        nGpuLayers: config.nGpuLayers,
        sampling: config.sampling,
      );

      final context = LlamaContext.create(model, inferenceConfig);
      logger.info('上下文创建完成');

      final chatTokenizer = LlamaTokenizer(model: model);
      logger.info('ChatTokenizer创建完成');

      onSuccess(model, context, chatTokenizer);

      responsePort.send(_IsolateMessage(_MessageType.loadModelResult, {'success': true}));
    } catch (e) {
      logger.error('加载模型失败: $e');
      responsePort.send(_IsolateMessage(_MessageType.error, e.toString()));
    }
  }

  static Future<void> _handleGenerateAsync(
    Map<String, dynamic> data,
    LlamaContext context,
    bool Function() shouldStop,
    Logger logger,
  ) async {
    final tokens = data['tokens'] as List<int>;
    final maxTokens = data['maxTokens'] as int;
    final responsePort = data['responsePort'] as SendPort;

    try {
      logger.debug('生成中: ${tokens.length}个Token, 最大$maxTokens个');

      context.resetPerformanceMetrics();

      final kvStatus = context.kvCacheStatus;
      if (kvStatus.isFull) {
        logger.error('生成前KV缓存已满');
        responsePort.send(_IsolateMessage(_MessageType.error, 'KV缓存已满，需要重置'));
        return;
      }

      context.decode(tokens);

      for (var i = 0; i < maxTokens; i++) {
        if (shouldStop()) {
          logger.debug('用户停止生成');
          responsePort.send(_IsolateMessage(_MessageType.stopped, null));
          return;
        }

        final currentKvStatus = context.kvCacheStatus;
        if (currentKvStatus.isFull) {
          logger.error('生成过程中KV缓存已满，步骤$i');
          responsePort.send(_IsolateMessage(_MessageType.error, 'KV缓存已满，生成中断'));
          return;
        }

        if (currentKvStatus.isOverSafeThreshold) {
          logger.warning('KV缓存超过安全阈值: ${currentKvStatus.usagePercent.toStringAsFixed(1)}%');
        }

        final token = context.sample();
        final isEnd = context.isEos(token);
        final tokenText = context.isControl(token) ? '' : context.detokenizeOne(token);

        final updatedKvStatus = context.kvCacheStatus;
        responsePort.send(
          _IsolateMessage(
            _MessageType.tokenGeneration,
            TokenGeneration(
              token: token,
              text: tokenText,
              isEnd: isEnd,
              kvUsed: updatedKvStatus.used,
              kvTotal: updatedKvStatus.total,
              kvUsagePercent: updatedKvStatus.usagePercent,
            ),
          ),
        );

        if (isEnd) break;

        context.decodeOne(token);

        await Future.delayed(Duration.zero);
      }

      responsePort.send(_IsolateMessage(_MessageType.done, null));
    } catch (e) {
      logger.error('生成错误: $e');
      responsePort.send(_IsolateMessage(_MessageType.error, e.toString()));
    }
  }

  static void _handleTokenize(Map<String, dynamic> data, LlamaModel model, Logger logger) {
    final text = data['text'] as String;
    final addBos = data['addBos'] as bool;
    final responsePort = data['responsePort'] as SendPort;

    try {
      final tokens = model.tokenize(text, addBos: addBos);
      responsePort.send(_IsolateMessage(_MessageType.tokenizeResult, tokens));
    } catch (e) {
      logger.error('Tokenize错误: $e');
      responsePort.send(_IsolateMessage(_MessageType.error, e.toString()));
    }
  }

  static void _handleApplyChatTemplate(Map<String, dynamic> data, LlamaTokenizer chatTokenizer, Logger logger) {
    final messages = data['messages'] as List<Map<String, String>>;
    final responsePort = data['responsePort'] as SendPort;

    try {
      final chatMessages = messages.map((m) {
        final role = m['role']!;
        final content = m['content']!;
        return LlamaChatMessage(
          role: role == 'system'
              ? LlamaMessageRole.system
              : role == 'user'
              ? LlamaMessageRole.user
              : LlamaMessageRole.assistant,
          content: content,
        );
      }).toList();

      final formattedText = chatTokenizer.applyTemplate(chatMessages);
      logger.debug('应用Chat模板，长度: ${formattedText.length}');

      responsePort.send(_IsolateMessage(_MessageType.applyChatTemplateResult, formattedText));
    } catch (e) {
      logger.error('应用Chat模板错误: $e');
      responsePort.send(_IsolateMessage(_MessageType.error, e.toString()));
    }
  }

  static void _handleSetKeepPrefixTokens(Map<String, dynamic> data, LlamaContext? context, Logger logger) {
    final responsePort = data['responsePort'] as SendPort;
    final tokens = List<int>.from(data['tokens'] as List);

    try {
      context?.setKeepPrefixTokens(tokens);
      logger.info('设置保留前缀Token: ${tokens.length}个');
      responsePort.send(_IsolateMessage(_MessageType.setKeepPrefixTokensResult, {'success': true}));
    } catch (e) {
      logger.error('设置保留前缀Token错误: $e');
      responsePort.send(_IsolateMessage(_MessageType.error, e.toString()));
    }
  }

  static void _handleReset(Map<String, dynamic> data, LlamaContext? context, Logger logger) {
    final responsePort = data['responsePort'] as SendPort;

    try {
      context?.reset();
      responsePort.send(_IsolateMessage(_MessageType.resetResult, null));
    } catch (e) {
      logger.error('重置错误: $e');
      responsePort.send(_IsolateMessage(_MessageType.error, e.toString()));
    }
  }

  static void _handleDispose(
    Map<String, dynamic> data,
    LlamaModel? model,
    LlamaContext? context,
    Function() onDisposed,
    Logger logger,
  ) {
    final responsePort = data['responsePort'] as SendPort;

    try {
      context?.dispose();
      model?.dispose();
      onDisposed();
      responsePort.send(_IsolateMessage(_MessageType.disposeResult, null));
    } catch (e) {
      logger.error('释放错误: $e');
      responsePort.send(_IsolateMessage(_MessageType.error, e.toString()));
    }
  }

  static void _handleGetPerformanceMetrics(Map<String, dynamic> data, LlamaContext? context, Logger logger) {
    final responsePort = data['responsePort'] as SendPort;

    try {
      final metrics = context?.performanceMetrics ?? PerformanceMetrics.empty();
      responsePort.send(
        _IsolateMessage(_MessageType.performanceMetricsResult, {
          'tStartMs': metrics.tStartMs,
          'tLoadMs': metrics.tLoadMs,
          'tPromptEvalMs': metrics.tPromptEvalMs,
          'tEvalMs': metrics.tEvalMs,
          'nPromptEval': metrics.nPromptEval,
          'nEval': metrics.nEval,
          'nReused': metrics.nReused,
          'tSampleMs': metrics.tSampleMs,
          'nSample': metrics.nSample,
        }),
      );
    } catch (e) {
      logger.error('获取性能指标错误: $e');
      responsePort.send(_IsolateMessage(_MessageType.error, e.toString()));
    }
  }

  static void _handleEmbed(Map<String, dynamic> data, LlamaModel? model, Logger logger) {
    final responsePort = data['responsePort'] as SendPort;
    final text = data['text'] as String;

    try {
      if (model == null) {
        throw StateError('模型未加载');
      }

      final tokens = model.tokenize(text, addBos: true);
      if (tokens.isEmpty) {
        responsePort.send(_IsolateMessage(_MessageType.embedResult, <double>[]));
        return;
      }

      final embedding = model.embed(tokens);
      responsePort.send(_IsolateMessage(_MessageType.embedResult, embedding));
    } catch (e) {
      logger.error('嵌入错误: $e');
      responsePort.send(_IsolateMessage(_MessageType.error, e.toString()));
    }
  }

  static void _handlePauseKVCache(Map<String, dynamic> data, LlamaContext? context, Logger logger) {
    final responsePort = data['responsePort'] as SendPort;

    try {
      context?.pauseKVCache();
      logger.info('KV Cache 已暂停');
      responsePort.send(_IsolateMessage(_MessageType.pauseKVCacheResult, {'success': true}));
    } catch (e) {
      logger.error('暂停 KV Cache 错误: $e');
      responsePort.send(_IsolateMessage(_MessageType.error, e.toString()));
    }
  }

  static void _handleResumeKVCache(Map<String, dynamic> data, LlamaContext? context, Logger logger) {
    final responsePort = data['responsePort'] as SendPort;

    try {
      context?.resumeKVCache();
      logger.info('KV Cache 已恢复');
      responsePort.send(_IsolateMessage(_MessageType.resumeKVCacheResult, {'success': true}));
    } catch (e) {
      logger.error('恢复 KV Cache 错误: $e');
      responsePort.send(_IsolateMessage(_MessageType.error, e.toString()));
    }
  }

  static void _handlePrepareForSnapshot(Map<String, dynamic> data, LlamaContext? context, Logger logger) {
    final responsePort = data['responsePort'] as SendPort;

    try {
      context?.prepareForSnapshot();
      logger.info('已准备接收快照');
      responsePort.send(_IsolateMessage(_MessageType.prepareForSnapshotResult, {'success': true}));
    } catch (e) {
      logger.error('准备快照错误: $e');
      responsePort.send(_IsolateMessage(_MessageType.error, e.toString()));
    }
  }

  static void _handleInjectContextTokens(Map<String, dynamic> data, LlamaContext? context, Logger logger) {
    final responsePort = data['responsePort'] as SendPort;
    final tokens = List<int>.from(data['tokens'] as List);

    try {
      context?.injectContextTokens(tokens);
      logger.info('注入上下文 Token: ${tokens.length} 个');
      responsePort.send(_IsolateMessage(_MessageType.injectContextTokensResult, {'success': true}));
    } catch (e) {
      logger.error('注入上下文 Token 错误: $e');
      responsePort.send(_IsolateMessage(_MessageType.error, e.toString()));
    }
  }
}
