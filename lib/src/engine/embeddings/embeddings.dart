import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:llama_native/src/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/engine/model/llama_model.dart';
import 'package:llama_native/src/engine/exceptions/llama_exceptions.dart';
import 'package:llama_native/src/log/logger.dart';
import 'package:llama_native/src/utils/disposable.dart';

enum PoolingType {
  none(0),
  mean(1),
  cls(2),
  last(3),
  rank(4);

  final int value;
  const PoolingType(this.value);

  static PoolingType fromValue(int value) => switch (value) {
    0 => PoolingType.none,
    1 => PoolingType.mean,
    2 => PoolingType.cls,
    3 => PoolingType.last,
    4 => PoolingType.rank,
    _ => throw ArgumentError('Unknown PoolingType: $value'),
  };
}

class EmbeddingResult {
  final List<double> embedding;
  final int tokenCount;
  final int embeddingDim;

  EmbeddingResult({required this.embedding, required this.tokenCount, required this.embeddingDim});

  double similarity(EmbeddingResult other) {
    if (embedding.length != other.embedding.length) {
      throw ArgumentError('Embedding dimensions do not match');
    }

    double dotProduct = 0;
    double normA = 0;
    double normB = 0;

    for (int i = 0; i < embedding.length; i++) {
      dotProduct += embedding[i] * other.embedding[i];
      normA += embedding[i] * embedding[i];
      normB += other.embedding[i] * other.embedding[i];
    }

    if (normA == 0 || normB == 0) return 0;
    return dotProduct / (sqrt(normA) * sqrt(normB));
  }

  double sqrt(double x) {
    if (x <= 0) return 0;
    double guess = x / 2;
    for (int i = 0; i < 20; i++) {
      guess = (guess + x / guess) / 2;
    }
    return guess;
  }

  Float32List toFloat32() => Float32List.fromList(embedding);

  @override
  String toString() => 'EmbeddingResult(dim: $embeddingDim, tokens: $tokenCount)';
}

class EmbeddingsConfig {
  final PoolingType poolingType;
  final int nThreads;
  final bool normalize;
  final int maxBatchSize;

  const EmbeddingsConfig({
    this.poolingType = PoolingType.mean,
    this.nThreads = 4,
    this.normalize = true,
    this.maxBatchSize = 512,
  });
}

class LlamaEmbeddings with Disposable {
  final LlamaModel _model;
  final EmbeddingsConfig _config;
  final Logger _logger;

  Pointer<bindings.llama_context>? _ctx;
  bool _disposed = false;

  LlamaEmbeddings._(this._model, this._config) : _logger = Logger('LlamaEmbeddings');

  static Future<LlamaEmbeddings> create(LlamaModel model, {EmbeddingsConfig config = const EmbeddingsConfig()}) async {
    final embeddings = LlamaEmbeddings._(model, config);
    await embeddings._initialize();
    return embeddings;
  }

  Future<void> _initialize() async {
    _logger.info('Initializing embeddings context');

    final ctxParams = bindings.llama_context_default_params();
    ctxParams.n_ctx = 2048;
    ctxParams.n_batch = _config.maxBatchSize;
    ctxParams.n_ubatch = _config.maxBatchSize;
    ctxParams.n_threads = _config.nThreads;
    ctxParams.n_threads_batch = _config.nThreads;
    ctxParams.embeddings = true;
    ctxParams.pooling_typeAsInt = _config.poolingType.value;

    _ctx = bindings.llama_new_context_with_model(_model.handle, ctxParams);

    if (_ctx == null || _ctx == nullptr) {
      throw LlamaException.context('Failed to create embeddings context');
    }

    _logger.info('Embeddings context initialized');
  }

  EmbeddingResult embed(String text) {
    if (_disposed) throw StateError('Embeddings is disposed');

    final tokens = _tokenize(text);
    return _computeEmbedding(tokens);
  }

  List<EmbeddingResult> embedBatch(List<String> texts) {
    if (_disposed) throw StateError('Embeddings is disposed');

    return texts.map((text) => embed(text)).toList();
  }

  EmbeddingResult embedTokens(List<int> tokens) {
    if (_disposed) throw StateError('Embeddings is disposed');
    return _computeEmbedding(tokens);
  }

  List<int> _tokenize(String text) {
    final textC = text.toNativeUtf8().cast<Char>();
    final tokens = calloc<bindings.llama_token>(text.length + 2);

    try {
      final nTokens = bindings.llama_tokenize(_model.vocab, textC, text.length, tokens, text.length + 2, true, true);

      if (nTokens < 0) {
        throw LlamaException.tokenize('Failed to tokenize text');
      }

      return List.generate(nTokens, (i) => tokens[i]);
    } finally {
      calloc.free(textC);
      calloc.free(tokens);
    }
  }

  EmbeddingResult _computeEmbedding(List<int> tokens) {
    if (tokens.isEmpty) {
      throw LlamaException.inference('Cannot embed empty text');
    }

    final batch = bindings.llama_batch_init(tokens.length, 0, 1);

    try {
      for (int i = 0; i < tokens.length; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i].value = 0;
        batch.logits[i] = 0;
      }
      batch.n_tokens = tokens.length;

      bindings.llama_set_embeddings(_ctx!, true);
      bindings.llama_set_causal_attn(_ctx!, false);

      final result = bindings.llama_encode(_ctx!, batch);
      if (result != 0) {
        throw LlamaException.inference('Failed to encode batch');
      }

      final embeddingsPtr = bindings.llama_get_embeddings_seq(_ctx!, 0);
      if (embeddingsPtr == nullptr) {
        throw LlamaException.inference('Failed to get embeddings');
      }

      final metadata = _model.getMetadata();
      final embeddingDim = metadata.embeddingDimension;
      final embedding = <double>[];

      for (int i = 0; i < embeddingDim; i++) {
        embedding.add(embeddingsPtr[i]);
      }

      if (_config.normalize) {
        _normalize(embedding);
      }

      return EmbeddingResult(embedding: embedding, tokenCount: tokens.length, embeddingDim: embeddingDim);
    } finally {
      bindings.llama_batch_free(batch);
    }
  }

  void _normalize(List<double> embedding) {
    double norm = 0;
    for (final val in embedding) {
      norm += val * val;
    }
    norm = sqrt(norm);

    if (norm > 0) {
      for (int i = 0; i < embedding.length; i++) {
        embedding[i] /= norm;
      }
    }
  }

  double sqrt(double x) {
    if (x <= 0) return 0;
    double guess = x / 2;
    for (int i = 0; i < 20; i++) {
      guess = (guess + x / guess) / 2;
    }
    return guess;
  }

  int get embeddingSize => _model.getMetadata().embeddingDimension;

  PoolingType get poolingType {
    if (_ctx == null) return PoolingType.none;
    return PoolingType.fromValue(bindings.llama_pooling_type$1(_ctx!).value);
  }

  @override
  bool get isDisposed => _disposed;

  @override
  void dispose() {
    if (_disposed) return;

    _logger.debug('Disposing embeddings context');

    if (_ctx != null && _ctx != nullptr) {
      bindings.llama_free(_ctx!);
      _ctx = null;
    }

    _disposed = true;
    _logger.info('Embeddings disposed');
  }
}
