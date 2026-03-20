import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:llama_native/src/llama_native_bindings.dart' as bindings;
import 'package:llama_native/src/engine/model/llama_model.dart';
import 'package:llama_native/src/engine/exceptions/llama_exceptions.dart';
import 'package:llama_native/src/log/logger.dart';

enum MediaType { image, audio }

class MediaContent {
  final MediaType type;
  final Uint8List data;
  final int width;
  final int height;
  final String? id;

  MediaContent.image(this.data, {int? width, int? height, this.id})
    : type = MediaType.image,
      width = width ?? 0,
      height = height ?? 0;

  MediaContent.audio(this.data, {this.id}) : type = MediaType.audio, width = 0, height = 0;

  bool get isImage => type == MediaType.image;
  bool get isAudio => type == MediaType.audio;
}

class MultimodalConfig {
  final String? mmprojPath;
  final bool useGpu;
  final int nThreads;
  final bool printTimings;
  final String mediaMarker;
  final bool warmup;

  const MultimodalConfig({
    this.mmprojPath,
    this.useGpu = true,
    this.nThreads = 4,
    this.printTimings = false,
    this.mediaMarker = '<__media__>',
    this.warmup = true,
  });

  bool get hasMmproj => mmprojPath != null && mmprojPath!.isNotEmpty;
}

class MultimodalCapabilities {
  final bool supportsVision;
  final bool supportsAudio;
  final int audioSampleRate;

  const MultimodalCapabilities({this.supportsVision = false, this.supportsAudio = false, this.audioSampleRate = -1});
}

abstract class MultimodalProcessor {
  MultimodalCapabilities get capabilities;

  Future<void> initialize(LlamaModel model, MultimodalConfig config);
  Future<List<int>> processMedia(MediaContent media);
  Future<List<int>> processPromptWithMedia(String prompt, List<MediaContent> media);
  void reset();
  void dispose();
}

class MultimodalProcessorImpl implements MultimodalProcessor {
  final Logger _logger = Logger('MultimodalProcessor');

  LlamaModel? _model;
  MultimodalConfig? _config;
  MultimodalCapabilities _capabilities = const MultimodalCapabilities();

  @override
  MultimodalCapabilities get capabilities => _capabilities;

  @override
  Future<void> initialize(LlamaModel model, MultimodalConfig config) async {
    _model = model;
    _config = config;

    if (!config.hasMmproj) {
      _logger.warning('No mmproj path provided, multimodal support disabled');
      _capabilities = const MultimodalCapabilities();
      return;
    }

    _logger.info('Initializing multimodal processor with mmproj: ${config.mmprojPath}');

    _capabilities = const MultimodalCapabilities(supportsVision: true, supportsAudio: false);

    _logger.info('Multimodal processor initialized');
  }

  @override
  Future<List<int>> processMedia(MediaContent media) async {
    if (_model == null) {
      throw StateError('Multimodal processor not initialized');
    }

    _logger.debug('Processing ${media.type} media');

    if (media.isImage) {
      return _processImage(media.data, media.width, media.height);
    } else if (media.isAudio) {
      return _processAudio(media.data);
    }

    throw LlamaException.inference('Unsupported media type: ${media.type}');
  }

  Future<List<int>> _processImage(Uint8List imageData, int? width, int? height) async {
    _logger.debug('Processing image: ${imageData.length} bytes');

    final tokens = <int>[];

    final imageTokenStart = _findSpecialToken('<|image|>');
    final imageTokenEnd = _findSpecialToken('<|/image|>');

    if (imageTokenStart >= 0) {
      tokens.add(imageTokenStart);
    }

    final placeholderCount = 256;
    for (int i = 0; i < placeholderCount; i++) {
      tokens.add(0);
    }

    if (imageTokenEnd >= 0) {
      tokens.add(imageTokenEnd);
    }

    return tokens;
  }

  Future<List<int>> _processAudio(Uint8List audioData) async {
    _logger.debug('Processing audio: ${audioData.length} bytes');

    if (!_capabilities.supportsAudio) {
      throw LlamaException.inference('Audio processing not supported');
    }

    return [];
  }

  int _findSpecialToken(String token) {
    if (_model == null) return -1;

    final vocab = _model!.vocab;
    final tokenC = token.toNativeUtf8().cast<Char>();
    final tokens = calloc<bindings.llama_token>(1);

    try {
      final nTokens = bindings.llama_tokenize(vocab, tokenC, token.length, tokens, 1, false, true);
      if (nTokens == 1) {
        return tokens[0];
      }
      return -1;
    } catch (e) {
      return -1;
    } finally {
      calloc.free(tokenC);
      calloc.free(tokens);
    }
  }

  @override
  Future<List<int>> processPromptWithMedia(String prompt, List<MediaContent> media) async {
    if (media.isEmpty) {
      return _tokenizeText(prompt);
    }

    final allTokens = <int>[];
    final marker = _config?.mediaMarker ?? '<__media__>';
    final parts = prompt.split(marker);

    for (int i = 0; i < parts.length; i++) {
      if (parts[i].isNotEmpty) {
        allTokens.addAll(await _tokenizeText(parts[i]));
      }

      if (i < media.length && i < parts.length - 1) {
        final mediaTokens = await processMedia(media[i]);
        allTokens.addAll(mediaTokens);
      }
    }

    return allTokens;
  }

  Future<List<int>> _tokenizeText(String text) async {
    if (_model == null) return [];

    final textC = text.toNativeUtf8().cast<Char>();
    final tokens = calloc<bindings.llama_token>(text.length + 2);

    try {
      final nTokens = bindings.llama_tokenize(_model!.vocab, textC, text.length, tokens, text.length + 2, true, true);

      if (nTokens < 0) return [];

      return List.generate(nTokens, (i) => tokens[i]);
    } finally {
      calloc.free(textC);
      calloc.free(tokens);
    }
  }

  @override
  void reset() {
    _logger.debug('Resetting multimodal processor');
  }

  @override
  void dispose() {
    if (_model == null) return;

    _logger.debug('Disposing multimodal processor');
    reset();

    _model = null;
    _config = null;
  }
}

class LlamaVision {
  final LlamaModel _model;
  final MultimodalProcessor _processor;
  final Logger _logger = Logger('LlamaVision');

  LlamaVision._(this._model, this._processor);

  static Future<LlamaVision> create(LlamaModel model, {MultimodalConfig config = const MultimodalConfig()}) async {
    final processor = MultimodalProcessorImpl();
    await processor.initialize(model, config);
    return LlamaVision._(model, processor);
  }

  bool get supportsVision => _processor.capabilities.supportsVision;
  bool get supportsAudio => _processor.capabilities.supportsAudio;

  Future<List<int>> encodeImage(Uint8List imageData, {int? width, int? height}) async {
    if (!supportsVision) {
      throw LlamaException.inference('Vision not supported');
    }

    final media = MediaContent.image(imageData, width: width, height: height);
    return _processor.processMedia(media);
  }

  Future<List<int>> encodeAudio(Uint8List audioData) async {
    if (!supportsAudio) {
      throw LlamaException.inference('Audio not supported');
    }

    final media = MediaContent.audio(audioData);
    return _processor.processMedia(media);
  }

  Future<List<int>> prepareMultimodalPrompt(String prompt, List<MediaContent> media) async {
    return _processor.processPromptWithMedia(prompt, media);
  }

  void reset() {
    _processor.reset();
  }

  void dispose() {
    _processor.dispose();
  }
}

class ImageUtils {
  static Uint8List? resizeImage(Uint8List imageData, int targetWidth, int targetHeight) {
    return null;
  }

  static (int, int)? getImageDimensions(Uint8List imageData) {
    return null;
  }

  static Uint8List? convertToRGB(Uint8List imageData) {
    return null;
  }

  static Uint8List? loadFromFile(String path) {
    return null;
  }
}
