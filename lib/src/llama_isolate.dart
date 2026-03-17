import 'dart:async';
import 'dart:isolate';

import 'package:llama_native/llama_chat_message.dart';
import 'package:llama_native/src/engine/context/inference_config.dart';
import 'package:llama_native/src/engine/context/llama_context.dart';
import 'package:llama_native/src/engine/context/token_generation.dart';
import 'package:llama_native/src/engine/model/llama_model.dart';
import 'package:llama_native/src/engine/sampling/sampling_config.dart';
import 'package:llama_native/src/engine/tokenizer/llama_tokenizer.dart';
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
  bool _isInitialized = false;
  bool _isModelLoaded = false;
  bool _isGenerating = false;
  SendPort? _currentGeneratePort;

  bool get isInitialized => _isInitialized;
  bool get isModelLoaded => _isModelLoaded;
  bool get isGenerating => _isGenerating;

  Future<void> initialize() async {
    if (_isInitialized) return;

    _receivePort = ReceivePort();
    _isolate = await Isolate.spawn(_isolateEntryPoint, _receivePort!.sendPort, debugName: 'LlamaIsolate');

    final completer = Completer<void>();
    _subscription = _receivePort!.listen((message) {
      if (message is SendPort) {
        _sendPort = message;
        _isInitialized = true;
        completer.complete();
      } else if (message is _IsolateMessage) {
        _handleMessage(message);
      }
    });

    await completer.future;
    _logger.info('LlamaIsolate initialized');
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
    if (!_isInitialized || _sendPort == null) {
      throw StateError('Isolate not initialized');
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
      throw StateError('Model not loaded');
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
      throw StateError('Model not loaded');
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
      throw StateError('Model not loaded');
    }

    _isGenerating = true;
    final responsePort = ReceivePort();
    final controller = StreamController<TokenGeneration>();
    _currentGeneratePort = responsePort.sendPort;

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
            _currentGeneratePort = null;
            controller.close();
            responsePort.close();
            break;
          case _MessageType.error:
            _isGenerating = false;
            _currentGeneratePort = null;
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
    _isInitialized = false;
    _isModelLoaded = false;
    _isGenerating = false;
    _currentGeneratePort = null;

    _logger.info('LlamaIsolate disposed');
  }

  static void _isolateEntryPoint(SendPort mainSendPort) {
    final receivePort = ReceivePort();
    mainSendPort.send(receivePort.sendPort);

    final logger = Logger('LlamaIsolateWorker');

    LlamaModel? model;
    LlamaContext? context;
    LlamaTokenizer? chatTokenizer;
    bool shouldStop = false;
    SendPort? currentGeneratePort;

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
              currentGeneratePort = message.data['responsePort'] as SendPort;
              _handleGenerateAsync(message.data, context!, () => shouldStop, logger);
              currentGeneratePort = null;
            }
            break;

          case _MessageType.stop:
            shouldStop = true;
            logger.debug('Stop signal received');
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
      logger.info('Loading model: ${config.modelPath}');

      final model = LlamaModel.load(LlamaModelConfig(modelPath: config.modelPath));
      logger.info('Model loaded');

      final inferenceConfig = InferenceConfig(
        nCtx: config.nCtx,
        nBatch: config.nBatch,
        nUBatch: config.nUBatch,
        nThreads: config.nThreads,
        nGpuLayers: config.nGpuLayers,
        sampling: config.sampling,
      );

      final context = LlamaContext.create(model, inferenceConfig);
      logger.info('Context created');

      final chatTokenizer = LlamaTokenizer(model: model);
      logger.info('ChatTokenizer created');

      onSuccess(model, context, chatTokenizer);

      responsePort.send(_IsolateMessage(_MessageType.loadModelResult, {'success': true}));
    } catch (e) {
      logger.error('Failed to load model: $e');
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
      logger.debug('Generating with ${tokens.length} tokens, max $maxTokens');

      context.decode(tokens);

      for (var i = 0; i < maxTokens; i++) {
        if (shouldStop()) {
          logger.debug('Generation stopped by user');
          responsePort.send(_IsolateMessage(_MessageType.stopped, null));
          return;
        }

        final token = context.sample();
        final isEnd = context.isEos(token);
        final tokenText = context.detokenizeOne(token);

        responsePort.send(
          _IsolateMessage(_MessageType.tokenGeneration, TokenGeneration(token: token, text: tokenText, isEnd: isEnd)),
        );

        if (isEnd) break;

        context.decodeOne(token);

        await Future.delayed(Duration.zero);
      }

      responsePort.send(_IsolateMessage(_MessageType.done, null));
    } catch (e) {
      logger.error('Generation error: $e');
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
      logger.error('Tokenize error: $e');
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
      logger.debug('Applied chat template, length: ${formattedText.length}');

      responsePort.send(_IsolateMessage(_MessageType.applyChatTemplateResult, formattedText));
    } catch (e) {
      logger.error('Apply chat template error: $e');
      responsePort.send(_IsolateMessage(_MessageType.error, e.toString()));
    }
  }

  static void _handleReset(Map<String, dynamic> data, LlamaContext? context, Logger logger) {
    final responsePort = data['responsePort'] as SendPort;

    try {
      context?.reset();
      responsePort.send(_IsolateMessage(_MessageType.resetResult, null));
    } catch (e) {
      logger.error('Reset error: $e');
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
      logger.error('Dispose error: $e');
      responsePort.send(_IsolateMessage(_MessageType.error, e.toString()));
    }
  }
}
