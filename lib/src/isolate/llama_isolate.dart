import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:isolate';

import 'package:ffi/ffi.dart';
import 'package:llama_native/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/logging/logger.dart';
import 'package:llama_native/src/sampling/sampling_config.dart';

enum _MessageType {
  loadModel,
  loadModelResult,
  tokenize,
  tokenizeResult,
  generate,
  token,
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

  bool get isInitialized => _isInitialized;
  bool get isModelLoaded => _isModelLoaded;

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
      case _MessageType.token:
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

  Stream<String> generate(List<int> tokens, {int maxTokens = 1024}) async* {
    if (!_isModelLoaded || _sendPort == null) {
      throw StateError('Model not loaded');
    }

    final responsePort = ReceivePort();
    final controller = StreamController<String>();

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
          case _MessageType.token:
            controller.add(message.data as String);
            break;
          case _MessageType.done:
            controller.close();
            responsePort.close();
            break;
          case _MessageType.error:
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

    _logger.info('LlamaIsolate disposed');
  }

  static void _isolateEntryPoint(SendPort mainSendPort) {
    final receivePort = ReceivePort();
    mainSendPort.send(receivePort.sendPort);

    Pointer<bindings.llama_model>? modelPtr;
    Pointer<bindings.llama_context>? ctxPtr;
    Pointer<bindings.llama_sampler>? samplerPtr;
    Pointer<bindings.llama_vocab>? vocabPtr;
    int nPast = 0;
    int nCtx = 4096;
    int nUBatch = 128;

    bindings.llama_backend_init();

    receivePort.listen((message) {
      if (message is _IsolateMessage) {
        switch (message.type) {
          case _MessageType.loadModel:
            _handleLoadModel(message.data, (model, ctx, sampler, vocab, ctxSize, uBatchSize) {
              modelPtr = model;
              ctxPtr = ctx;
              samplerPtr = sampler;
              vocabPtr = vocab;
              nCtx = ctxSize;
              nUBatch = uBatchSize;
              nPast = 0;
            });
            break;

          case _MessageType.generate:
            if (ctxPtr != null && samplerPtr != null && vocabPtr != null) {
              _handleGenerate(
                message.data,
                ctxPtr!,
                samplerPtr!,
                vocabPtr!,
                modelPtr!,
                nCtx,
                nUBatch,
                (newNPast) => nPast = newNPast,
                nPast,
              );
            }
            break;

          case _MessageType.tokenize:
            if (vocabPtr != null) {
              _handleTokenize(message.data, vocabPtr!);
            }
            break;

          case _MessageType.reset:
            _handleReset(message.data, ctxPtr, samplerPtr, () => nPast = 0);
            break;

          case _MessageType.dispose:
            _handleDispose(message.data, modelPtr, ctxPtr, samplerPtr, () {
              modelPtr = null;
              ctxPtr = null;
              samplerPtr = null;
              vocabPtr = null;
            });
            break;

          default:
            break;
        }
      }
    });
  }

  static void _handleLoadModel(
    Map<String, dynamic> data,
    Function(
      Pointer<bindings.llama_model>,
      Pointer<bindings.llama_context>,
      Pointer<bindings.llama_sampler>,
      Pointer<bindings.llama_vocab>,
      int,
      int,
    )
    onSuccess,
  ) {
    final config = data['config'] as LlamaIsolateConfig;
    final responsePort = data['responsePort'] as SendPort;

    try {
      final modelParams = bindings.llama_model_default_params();
      modelParams.n_gpu_layers = config.nGpuLayers;
      modelParams.use_mmap = true;
      modelParams.use_mlock = false;

      final pathC = config.modelPath.toNativeUtf8().cast<Char>();
      final model = bindings.llama_load_model_from_file(pathC, modelParams);
      calloc.free(pathC);

      if (model == nullptr) {
        responsePort.send(_IsolateMessage(_MessageType.error, 'Failed to load model'));
        return;
      }

      final ctxParams = bindings.llama_context_default_params();
      ctxParams.n_ctx = config.nCtx;
      ctxParams.n_batch = config.nBatch;
      ctxParams.n_ubatch = config.nUBatch;
      ctxParams.n_threads = config.nThreads;
      ctxParams.n_threads_batch = config.nThreads;

      final ctx = bindings.llama_init_from_model(model, ctxParams);
      if (ctx == nullptr) {
        bindings.llama_free_model(model);
        responsePort.send(_IsolateMessage(_MessageType.error, 'Failed to create context'));
        return;
      }

      final vocab = bindings.llama_model_get_vocab(model);

      final sampler = _buildSamplerChain(config.sampling);
      if (sampler == nullptr) {
        bindings.llama_free(ctx);
        bindings.llama_free_model(model);
        responsePort.send(_IsolateMessage(_MessageType.error, 'Failed to create sampler'));
        return;
      }

      onSuccess(model, ctx, sampler, vocab, config.nCtx, config.nUBatch);

      responsePort.send(_IsolateMessage(_MessageType.loadModelResult, {'success': true}));
    } catch (e) {
      responsePort.send(_IsolateMessage(_MessageType.error, e.toString()));
    }
  }

  static Pointer<bindings.llama_sampler> _buildSamplerChain(SamplingConfig sampling) {
    final chainParams = bindings.llama_sampler_chain_default_params();
    final chain = bindings.llama_sampler_chain_init(chainParams);

    if (chain == nullptr) return nullptr;

    if (sampling.penaltyRepeat != 1.0 || sampling.frequencyPenalty != 0.0 || sampling.presencePenalty != 0.0) {
      bindings.llama_sampler_chain_add(
        chain,
        bindings.llama_sampler_init_penalties(
          sampling.penaltyLastN,
          sampling.penaltyRepeat,
          sampling.frequencyPenalty,
          sampling.presencePenalty,
        ),
      );
    }

    if (sampling.temperature <= 0.0) {
      bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_greedy());
    } else {
      if (sampling.topK > 0) {
        bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_top_k(sampling.topK));
      }
      if (sampling.minP > 0.0) {
        bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_min_p(sampling.minP, 1));
      }
      if (sampling.topP < 1.0) {
        bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_top_p(sampling.topP, 1));
      }
      bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_temp(sampling.temperature));
      bindings.llama_sampler_chain_add(chain, bindings.llama_sampler_init_dist(0xFFFFFFFF));
    }

    return chain;
  }

  static void _handleGenerate(
    Map<String, dynamic> data,
    Pointer<bindings.llama_context> ctx,
    Pointer<bindings.llama_sampler> sampler,
    Pointer<bindings.llama_vocab> vocab,
    Pointer<bindings.llama_model> model,
    int nCtx,
    int nUBatch,
    Function(int) updateNPast,
    int currentNPast,
  ) {
    final tokens = data['tokens'] as List<int>;
    final maxTokens = data['maxTokens'] as int;
    final responsePort = data['responsePort'] as SendPort;

    try {
      int nPast = currentNPast;

      if (tokens.isNotEmpty) {
        final batch = bindings.llama_batch_init(tokens.length, 0, 1);
        try {
          for (var i = 0; i < tokens.length; i++) {
            (batch.token + i).value = tokens[i];
            (batch.pos + i).value = nPast + i;
            (batch.n_seq_id + i).value = 1;
            ((batch.seq_id + i).value + 0).value = 0;
            (batch.logits + i).value = 0;
          }
          (batch.logits + tokens.length - 1).value = 1;
          batch.n_tokens = tokens.length;

          final ret = bindings.llama_decode(ctx, batch);
          if (ret < 0) {
            responsePort.send(_IsolateMessage(_MessageType.error, 'Decode failed: $ret'));
            return;
          } else if (ret > 0) {
            responsePort.send(_IsolateMessage(_MessageType.error, 'KV cache full'));
            return;
          }

          bindings.llama_synchronize(ctx);
          nPast += tokens.length;
        } finally {
          bindings.llama_batch_free(batch);
        }
      }

      updateNPast(nPast);

      for (var i = 0; i < maxTokens; i++) {
        final sampledToken = bindings.llama_sampler_sample(sampler, ctx, -1);
        bindings.llama_sampler_accept(sampler, sampledToken);

        final isEnd = bindings.llama_token_is_eog(vocab, sampledToken);
        nPast += 1;
        updateNPast(nPast);

        final tokenText = _detokenizeOne(vocab, sampledToken);
        responsePort.send(_IsolateMessage(_MessageType.token, tokenText));

        if (isEnd) break;

        final newBatch = bindings.llama_batch_init(1, 0, 1);
        try {
          (newBatch.token + 0).value = sampledToken;
          (newBatch.pos + 0).value = nPast - 1;
          (newBatch.n_seq_id + 0).value = 1;
          ((newBatch.seq_id + 0).value + 0).value = 0;
          (newBatch.logits + 0).value = 1;
          newBatch.n_tokens = 1;

          final ret = bindings.llama_decode(ctx, newBatch);
          if (ret != 0) {
            responsePort.send(_IsolateMessage(_MessageType.error, 'Decode failed during generation: $ret'));
            return;
          }

          bindings.llama_synchronize(ctx);
        } finally {
          bindings.llama_batch_free(newBatch);
        }
      }

      responsePort.send(_IsolateMessage(_MessageType.done, null));
    } catch (e) {
      responsePort.send(_IsolateMessage(_MessageType.error, e.toString()));
    }
  }

  static void _handleTokenize(Map<String, dynamic> data, Pointer<bindings.llama_vocab> vocab) {
    final text = data['text'] as String;
    final addBos = data['addBos'] as bool;
    final responsePort = data['responsePort'] as SendPort;

    try {
      final textC = text.toNativeUtf8();
      final textLen = text.length;

      var nTokens = bindings.llama_tokenize(vocab, textC.cast(), textLen, nullptr, 0, addBos, true);

      if (nTokens < 0) {
        nTokens = -nTokens;
      }

      final tokens = <int>[];
      if (nTokens > 0) {
        final tokenPtr = calloc<bindings.llama_token>(nTokens);
        try {
          final actualTokens = bindings.llama_tokenize(vocab, textC.cast(), textLen, tokenPtr, nTokens, addBos, true);

          if (actualTokens > 0) {
            for (var i = 0; i < actualTokens; i++) {
              tokens.add(tokenPtr[i]);
            }
          }
        } finally {
          calloc.free(tokenPtr);
        }
      }

      calloc.free(textC);
      responsePort.send(_IsolateMessage(_MessageType.tokenizeResult, tokens));
    } catch (e) {
      responsePort.send(_IsolateMessage(_MessageType.error, e.toString()));
    }
  }

  static String _detokenizeOne(Pointer<bindings.llama_vocab> vocab, int token) {
    var bufferSize = bindings.llama_token_to_piece(vocab, token, nullptr, 0, 0, true);

    if (bufferSize < 0) {
      bufferSize = -bufferSize;
    }

    if (bufferSize == 0) {
      return '';
    }

    final pieceBuffer = calloc<Char>(bufferSize);
    try {
      final actualSize = bindings.llama_token_to_piece(vocab, token, pieceBuffer, bufferSize, 0, true);

      if (actualSize > 0) {
        final allBytes = <int>[];
        for (var i = 0; i < actualSize; i++) {
          final byte = (pieceBuffer + i).value;
          allBytes.add(byte < 0 ? byte + 256 : byte);
        }
        try {
          return utf8.decode(allBytes, allowMalformed: true);
        } catch (e) {
          return String.fromCharCodes(allBytes);
        }
      }
      return '';
    } finally {
      calloc.free(pieceBuffer);
    }
  }

  static void _handleReset(
    Map<String, dynamic> data,
    Pointer<bindings.llama_context>? ctx,
    Pointer<bindings.llama_sampler>? sampler,
    Function() resetNPast,
  ) {
    final responsePort = data['responsePort'] as SendPort;

    try {
      if (ctx != null) {
        final mem = bindings.llama_get_memory(ctx);
        bindings.llama_memory_clear(mem, true);
      }
      if (sampler != null) {
        bindings.llama_sampler_reset(sampler);
      }
      resetNPast();

      responsePort.send(_IsolateMessage(_MessageType.resetResult, null));
    } catch (e) {
      responsePort.send(_IsolateMessage(_MessageType.error, e.toString()));
    }
  }

  static void _handleDispose(
    Map<String, dynamic> data,
    Pointer<bindings.llama_model>? model,
    Pointer<bindings.llama_context>? ctx,
    Pointer<bindings.llama_sampler>? sampler,
    Function() clearPointers,
  ) {
    final responsePort = data['responsePort'] as SendPort;

    try {
      if (sampler != null) {
        bindings.llama_sampler_free(sampler);
      }
      if (ctx != null) {
        bindings.llama_free(ctx);
      }
      if (model != null) {
        bindings.llama_free_model(model);
      }

      bindings.llama_backend_free();

      clearPointers();
      responsePort.send(_IsolateMessage(_MessageType.disposeResult, null));
    } catch (e) {
      responsePort.send(_IsolateMessage(_MessageType.error, e.toString()));
    }
  }
}
