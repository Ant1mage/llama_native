import 'dart:ffi';
import 'dart:io';
import 'dart:convert';
import 'package:ffi/ffi.dart';
import 'package:llama_native/src/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/log/logger.dart';

import 'package:llama_native/src/engine/exceptions/llama_exceptions.dart';

/// Llama 模型配置
class LlamaModelConfig {
  /// 模型文件路径
  final String modelPath;

  /// GPU层数
  final int gpuLayers;

  /// 仅加载词表 (用于 tokenization 任务)
  final bool vocabOnly;

  /// 分片模式 (多文件模型)
  final bindings.llama_split_mode splitMode;

  /// 主 GPU 索引
  final int mainGpu;

  const LlamaModelConfig({
    required this.modelPath,
    this.gpuLayers = 0,
    this.vocabOnly = false,
    this.splitMode = bindings.llama_split_mode.LLAMA_SPLIT_MODE_LAYER,
    this.mainGpu = 0,
  });
}

/// Llama 模型元数据信息
class LlamaModelMetadata {
  /// 模型名称
  final String name;

  /// 参数量 (十亿)
  final double parameterCount;

  /// 上下文长度
  final int contextLength;

  /// 嵌入维度
  final int embeddingDimension;

  /// 层数
  final int layerCount;

  /// 词表大小
  final int vocabSize;

  const LlamaModelMetadata({
    required this.name,
    required this.parameterCount,
    required this.contextLength,
    required this.embeddingDimension,
    required this.layerCount,
    required this.vocabSize,
  });
}

/// 静态资源管理器
///
/// 负责：
/// - GGUF 模型加载与验证
/// - 内存映射优化 (mmap)
/// - 模型元数据访问接口
class LlamaModel {
  final LlamaModelConfig _config;
  final Logger _logger;
  Pointer<bindings.llama_model>? _modelPtr;

  /// 私有构造函数
  LlamaModel._(this._config) : _logger = Logger('LlamaModel');

  /// 加载模型（同步）
  factory LlamaModel.load(LlamaModelConfig config) {
    final model = LlamaModel._(config);
    model._load();
    return model;
  }

  /// 在 Isolate 中加载模型的静态方法（用于异步加载）
  // ignore: unused_element
  static LlamaModel _loadModelInIsolate(LlamaModelConfig config) {
    final model = LlamaModel._(config);
    model._load();
    return model;
  }

  /// 内部加载逻辑
  void _load() {
    if (_modelPtr != null) {
      throw StateError('Model already loaded');
    }

    _logger.info('Loading model: ${_config.modelPath}');

    // 验证文件存在
    final modelFile = File(_config.modelPath);
    if (!modelFile.existsSync()) {
      throw FileSystemException('Model file not found', _config.modelPath);
    }

    // 验证文件权限
    try {
      final fileStat = modelFile.statSync();
      if (!fileStat.modeString().contains('r')) {
        throw FileSystemException('Model file is not readable', _config.modelPath);
      }
    } catch (e) {
      _logger.error('Error checking file permissions: $e');
      throw FileSystemException('Failed to access model file', _config.modelPath);
    }

    // 直接创建模型参数
    final modelParams = bindings.llama_model_default_params();
    modelParams.n_gpu_layers = _config.gpuLayers;
    modelParams.split_modeAsInt = _config.splitMode.value;
    modelParams.main_gpu = _config.mainGpu;
    modelParams.vocab_only = _config.vocabOnly;

    // 转换为 C 字符串
    final pathC = _config.modelPath.toNativeUtf8().cast<Char>();

    try {
      // 加载模型
      final ptr = bindings.llama_load_model_from_file(pathC, modelParams);

      if (ptr == nullptr) {
        _logger.error('Failed to load model: ${_config.modelPath}');
        throw LlamaException.model('Failed to load model from file', filePath: _config.modelPath);
      }

      _modelPtr = ptr;
      _logger.info('Model loaded successfully');

      // 打印模型信息
      try {
        final metadata = getMetadata();
        _logger.info('Model: ${metadata.name}');
        _logger.info('Parameters: ${metadata.parameterCount.toStringAsFixed(1)}B');
        _logger.info('Context: ${metadata.contextLength}, Vocab: ${metadata.vocabSize}');
      } catch (e) {
        _logger.warning('Failed to get model metadata: $e');
      }
    } catch (e) {
      _logger.error('Error loading model: $e');
      rethrow;
    } finally {
      calloc.free(pathC);
    }
  }

  /// 获取模型指针
  Pointer<bindings.llama_model> get handle {
    if (_modelPtr == null) {
      throw StateError('Model is disposed');
    }
    return _modelPtr!;
  }

  /// 获取词表指针
  Pointer<bindings.llama_vocab> get vocab {
    // 使用 llama_model_get_vocab 获取词表指针
    return bindings.llama_model_get_vocab(handle);
  }

  /// 获取模型元数据
  LlamaModelMetadata getMetadata() {
    final modelHandle = handle;
    final vocabHandle = vocab;

    // 获取基础信息
    final nVocab = bindings.llama_n_vocab(vocabHandle);
    final nCtxTrain = bindings.llama_n_ctx_train(modelHandle);
    final nEmbd = bindings.llama_n_embd(modelHandle);
    final nLayer = bindings.llama_n_layer(modelHandle);

    // 估算参数量 (简化计算)
    final paramCount = _estimateParameterCount(nEmbd, nLayer, nVocab);

    // 获取模型名称 (从 GGUF metadata)
    final name = _getModelName();

    return LlamaModelMetadata(
      name: name,
      parameterCount: paramCount,
      contextLength: nCtxTrain,
      embeddingDimension: nEmbd,
      layerCount: nLayer,
      vocabSize: nVocab,
    );
  }

  /// 估算参数量
  double _estimateParameterCount(int nEmbd, int nLayer, int nVocab) {
    // 简化估算：embedding + transformer layers
    final embeddingParams = nVocab * nEmbd;
    final transformerParams = nLayer * (nEmbd * nEmbd * 12); // 近似值
    final total = embeddingParams + transformerParams;
    return total / 1e9; // 转换为十亿
  }

  /// 获取模型名称
  String _getModelName() {
    // 尝试从文件名推断
    final fileName = _config.modelPath.split('/').last;

    // 常见模型命名模式
    final patterns = {'llama': 'LLaMA', 'qwen': 'Qwen', 'mistral': 'Mistral', 'gemma': 'Gemma', 'phi': 'Phi'};

    final lowerName = fileName.toLowerCase();
    for (final entry in patterns.entries) {
      if (lowerName.contains(entry.key)) {
        return '${entry.value} (${fileName.replaceAll('.gguf', '')})';
      }
    }

    return fileName.replaceAll('.gguf', '');
  }

  /// Tokenize 文本
  List<int> tokenize(String text, {bool addBos = true, bool addEos = false}) {
    if (_modelPtr == null) throw StateError('Model is disposed');
    if (text.isEmpty) return [];

    // 必须用 UTF-8 字节长度，Dart String.length 是 UTF-16 码元数，对多字节字符会偏小
    final textC = text.toNativeUtf8();
    final textByteLen = textC.length; // Utf8Pointer.length 返回字节数
    try {
      // 最大 token 数：UTF-8 字节数 + BOS/EOS 各1
      final maxTokens = textByteLen + (addBos ? 1 : 0) + (addEos ? 1 : 0) + 1;
      final tokens = calloc<Int32>(maxTokens);

      try {
        final n = bindings.llama_tokenize(
          vocab,
          textC.cast<Char>(),
          textByteLen,
          tokens,
          maxTokens,
          addBos,
          true, // parse_special: 解析特殊 token（如 <|im_start|> 等）
        );

        if (n < 0) {
          _logger.error('Tokenization failed with code: $n for text: $text');
          throw LlamaException.tokenize('Tokenization failed for text (code=$n)', text: text);
        }

        // 转换为 Dart List
        final result = <int>[];
        for (var i = 0; i < n; i++) {
          result.add(tokens.elementAt(i).value);
        }
        return result;
      } catch (e) {
        _logger.error('Error during tokenization: $e');
        throw LlamaException.tokenize('Tokenization failed: ${e.toString()}');
      } finally {
        calloc.free(tokens);
      }
    } catch (e) {
      _logger.error('Error in tokenize method: $e');
      rethrow;
    } finally {
      calloc.free(textC);
    }
  }

  /// Detokenize 回文本
  String detokenize(List<int> tokens) {
    if (_modelPtr == null) throw StateError('Model is disposed');
    if (tokens.isEmpty) return '';

    _logger.debug('detokenize: tokens=$tokens, vocab address=${vocab.address}');

    final allBytes = <int>[];

    for (final token in tokens) {
      try {
        var bufferSize = bindings.llama_token_to_piece(vocab, token, nullptr, 0, 0, true);

        if (bufferSize < 0) {
          bufferSize = -bufferSize;
        }

        if (bufferSize == 0) {
          continue;
        }

        final pieceBuffer = calloc<Char>(bufferSize);
        try {
          final actualSize = bindings.llama_token_to_piece(vocab, token, pieceBuffer, bufferSize, 0, true);

          if (actualSize > 0) {
            for (var i = 0; i < actualSize; i++) {
              final byte = pieceBuffer.elementAt(i).value;
              allBytes.add(byte < 0 ? byte + 256 : byte);
            }
          }
        } catch (e) {
          _logger.error('Error detokenizing token $token: $e');
        } finally {
          calloc.free(pieceBuffer);
        }
      } catch (e) {
        _logger.error('Error processing token $token: $e');
      }
    }

    try {
      return utf8.decode(allBytes, allowMalformed: true);
    } catch (e) {
      _logger.warning('UTF-8 decode failed: $e');
      return String.fromCharCodes(allBytes);
    }
  }

  List<double> embed(List<int> tokens) {
    if (_modelPtr == null) throw StateError('Model is disposed');
    if (tokens.isEmpty) return [];

    final nTokens = tokens.length;
    final ctxSize = nTokens > 512 ? nTokens : 512;

    final ctxParams = bindings.llama_context_default_params();
    ctxParams.n_ctx = ctxSize;
    ctxParams.n_batch = ctxSize;
    ctxParams.n_ubatch = ctxSize;
    ctxParams.embeddings = true;
    ctxParams.n_threads = 4;
    ctxParams.n_threads_batch = 4;

    final ctx = bindings.llama_new_context_with_model(_modelPtr!, ctxParams);
    if (ctx == nullptr) {
      throw LlamaException.context('Failed to create embedding context');
    }

    try {
      final batch = bindings.llama_batch_init(tokens.length, 0, 1);

      try {
        for (var i = 0; i < tokens.length; i++) {
          batch.token[i] = tokens[i];
          batch.pos[i] = i;
          batch.n_seq_id[i] = 1;
          batch.seq_id[i][0] = 0;
          batch.logits[i] = 0;
        }
        batch.n_tokens = tokens.length;
        batch.logits[tokens.length - 1] = 1;

        final result = bindings.llama_decode(ctx, batch);
        if (result != 0) {
          throw LlamaException.inference('Embedding decode failed: $result');
        }

        final nEmbd = bindings.llama_model_n_embd(_modelPtr!);
        final embeddingList = <double>[];

        var embeddings = bindings.llama_get_embeddings_seq(ctx, 0);
        if (embeddings != nullptr) {
          for (var i = 0; i < nEmbd; i++) {
            embeddingList.add(embeddings[i]);
          }
          return embeddingList;
        }

        embeddings = bindings.llama_get_embeddings_ith(ctx, tokens.length - 1);
        if (embeddings != nullptr) {
          for (var i = 0; i < nEmbd; i++) {
            embeddingList.add(embeddings[i]);
          }
          return embeddingList;
        }

        embeddings = bindings.llama_get_embeddings(ctx);
        if (embeddings != nullptr) {
          for (var i = 0; i < nEmbd; i++) {
            embeddingList.add(embeddings[i]);
          }
          return embeddingList;
        }

        throw LlamaException.inference('Failed to get embeddings from any method');
      } finally {
        bindings.llama_batch_free(batch);
      }
    } finally {
      bindings.llama_free(ctx);
    }
  }

  void dispose() {
    if (_modelPtr == null) return;

    _logger.info('Disposing model...');

    bindings.llama_free_model(_modelPtr!);
    _modelPtr = null;

    _logger.info('Model disposed');
  }
}
