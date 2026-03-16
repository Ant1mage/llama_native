import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'package:llama_native/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/backend/llama_backend.dart';
import 'package:llama_native/src/logging/logger.dart';
import 'package:llama_native/src/utils/disposable.dart';
import 'package:llama_native/src/exceptions/llama_exceptions.dart';

/// Llama 模型配置
class LlamaModelConfig {
  /// 模型文件路径
  final String modelPath;

  /// 仅加载词表 (用于 tokenization 任务)
  final bool vocabOnly;

  /// 分片模式 (多文件模型)
  final bindings.llama_split_mode splitMode;

  /// 主 GPU 索引
  final int mainGpu;

  const LlamaModelConfig({
    required this.modelPath,
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
class LlamaModel with Disposable {
  final LlamaModelConfig _config;
  final Logger _logger;
  Pointer<bindings.llama_model>? _modelPtr;
  bool _disposed = false;

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

    // 获取后端配置
    final backend = LlamaBackend.instance;
    final modelParams = backend.getModelParams();

    // 应用模型特定参数 (简化实现)
    // split_mode 需要在 backend 中设置

    // 转换为 C 字符串
    final pathC = _config.modelPath.toNativeUtf8().cast<Char>();

    try {
      // 加载模型
      final ptr = bindings.llama_load_model_from_file(pathC, modelParams);

      if (ptr == nullptr) {
        _logger.error('Failed to load model: ${_config.modelPath}');
        throw LlamaModelLoadException('Failed to load model from file', filePath: _config.modelPath);
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
    if (_modelPtr == null || _disposed) {
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

  @override
  bool get isDisposed => _disposed;

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
    if (_disposed) throw StateError('Model is disposed');
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
          throw LlamaTokenizeException('Tokenization failed for text (code=$n)', text: text);
        }

        // 转换为 Dart List
        final result = <int>[];
        for (var i = 0; i < n; i++) {
          result.add(tokens.elementAt(i).value);
        }
        return result;
      } catch (e) {
        _logger.error('Error during tokenization: $e');
        throw LlamaTokenizeException('Tokenization failed: ${e.toString()}');
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
    if (_disposed) throw StateError('Model is disposed');
    if (tokens.isEmpty) return '';

    final buffer = StringBuffer();

    for (final token in tokens) {
      try {
        // 先获取所需缓冲区大小
        final bufferSize = bindings.llama_token_to_piece(
          vocab,
          token,
          nullptr,
          0,
          0, // lstrip
          false, // special
        );

        _logger.debug('detokenize: token=$token, bufferSize=$bufferSize');

        if (bufferSize < 0) {
          _logger.debug('Invalid token: $token (bufferSize=$bufferSize)');
          continue;
        }

        if (bufferSize == 0) {
          continue;
        }

        // 分配缓冲区并获取文本
        final pieceBuffer = calloc<Char>(bufferSize);
        try {
          final actualSize = bindings.llama_token_to_piece(
            vocab,
            token,
            pieceBuffer,
            bufferSize,
            0, // lstrip
            false, // special
          );

          if (actualSize > 0) {
            // 转换为 Dart 字符串
            final stringBytes = Uint8List.fromList(List.generate(actualSize, (i) => pieceBuffer.elementAt(i).value));
            buffer.write(String.fromCharCodes(stringBytes));
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

    return buffer.toString();
  }

  @override
  void dispose() {
    if (_disposed) return;

    _logger.info('Disposing model...');

    if (_modelPtr != null) {
      bindings.llama_free_model(_modelPtr!);
      _modelPtr = null;
    }

    _disposed = true;
    _logger.info('Model disposed');
  }
}
