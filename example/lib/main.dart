import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'package:llama_native/llama_native_bindings.dart';

/// 解码单个 token
String _decodeToken(Pointer<llama_vocab> vocab, int token) {
  final buffer = calloc<Char>(256);

  try {
    final nChars = llama_token_to_piece(vocab, token, buffer, 256, 0, true);
    if (nChars < 0) {
      return '';
    }

    final result = buffer.cast<Utf8>().toDartString(length: nChars);
    return result;
  } finally {
    calloc.free(buffer);
  }
}

void main(List<String> args) async {
  print('🦙 LLaMA Native - 流式对话示例');
  print('=' * 50);

  // 1. 初始化后端
  print('\n📦 初始化 llama 后端...');
  llama_backend_init();

  try {
    // 2. 加载模型
    final modelPath = '../models/Qwen3.5-4B-Q4_K_M.gguf';
    print('\n📂 加载模型：$modelPath');

    if (!File(modelPath).existsSync()) {
      print('❌ 模型文件不存在：$modelPath');
      exit(1);
    }

    final modelPtr = _loadModel(modelPath);
    if (modelPtr == nullptr) {
      print('❌ 加载模型失败');
      exit(1);
    }
    print('✅ 模型加载成功');

    // 3. 创建上下文
    print('\n🔧 创建推理上下文...');
    final ctxPtr = _createContext(modelPtr);
    if (ctxPtr == nullptr) {
      print('❌ 创建上下文失败');
      llama_free_model(modelPtr);
      exit(1);
    }
    print('✅ 上下文创建成功');

    // 4. tokenize 输入
    final prompt = "你是谁？";
    print('\n💬 用户：$prompt');

    final tokens = _tokenize(ctxPtr, prompt);
    if (tokens.isEmpty) {
      print('❌ Tokenization 失败');
      llama_free(ctxPtr);
      llama_free_model(modelPtr);
      exit(1);
    }

    // 5. 生成回复（流式）
    print('\n🤖 助手：');
    await _generateStreamed(ctxPtr, tokens);

    // 6. 清理资源
    print('\n\n🧹 清理资源...');
    llama_free(ctxPtr);
    llama_free_model(modelPtr);
    print('✅ 资源清理完成');
  } catch (e) {
    print('❌ 错误：$e');
    llama_backend_free();
    exit(1);
  }

  // 7. 释放后端
  print('\n👋 释放后端...');
  llama_backend_free();
  print('✅ 程序结束');
}

/// 加载模型
Pointer<llama_model> _loadModel(String path) {
  final pathC = path.toNativeUtf8().cast<Char>();

  try {
    final modelParams = llama_model_default_params();
    return llama_load_model_from_file(pathC, modelParams);
  } finally {
    calloc.free(pathC);
  }
}

/// 创建上下文
Pointer<llama_context> _createContext(Pointer<llama_model> model) {
  final ctxParams = llama_context_default_params();
  // 设置上下文大小（根据需要调整）
  ctxParams.n_ctx = 2048;
  ctxParams.n_batch = 512;
  ctxParams.n_threads = Platform.numberOfProcessors;
  ctxParams.n_threads_batch = Platform.numberOfProcessors;

  return llama_new_context_with_model(model, ctxParams);
}

/// Tokenize 文本
List<int> _tokenize(Pointer<llama_context> ctx, String text) {
  final textC = text.toNativeUtf8().cast<Char>();

  // 获取 vocab
  final modelPtr = llama_get_model(ctx);
  final model = modelPtr.cast<llama_model>();
  final vocab = llama_model_get_vocab(model);

  try {
    // 首先获取需要的 token 数量
    final maxTokens = text.length + 256;
    final tokens = calloc<llama_token>(maxTokens);

    try {
      final nTokens = llama_tokenize(
        vocab,
        textC,
        text.length,
        tokens,
        maxTokens,
        true, // add_special
        true, // parse_special
      );

      if (nTokens < 0) {
        return [];
      }

      // 转换为 Dart List
      final result = <int>[];
      for (var i = 0; i < nTokens; i++) {
        result.add(tokens[i]);
      }
      return result;
    } finally {
      calloc.free(tokens);
    }
  } finally {
    calloc.free(textC);
  }
}

/// 流式生成文本
Future<void> _generateStreamed(Pointer<llama_context> ctx, List<int> inputTokens) async {
  final modelPtr = llama_get_model(ctx);
  final model = modelPtr.cast<llama_model>();
  final vocab = llama_model_get_vocab(model);

  // 创建 batch
  final batchSize = 512;
  final batch = llama_batch_init(batchSize, 0, 0);

  try {
    // 解码所有输入 tokens
    var nPast = 0;
    for (var i = 0; i < inputTokens.length; i += batchSize) {
      final end = (i + batchSize > inputTokens.length) ? inputTokens.length : i + batchSize;
      final nBatch = end - i;

      // 填充 batch
      for (var j = 0; j < nBatch; j++) {
        batch.token[i + j] = inputTokens[i + j];
        batch.pos[i + j] = nPast + j;
        batch.n_seq_id[i + j] = 1;
        batch.seq_id[i + j][0] = 0;
        batch.logits[i + j] = 0;
      }
      batch.n_tokens = nBatch;
      batch.logits[nBatch - 1] = 1; // 最后一个 token 需要 logits

      // 解码
      final result = llama_decode(ctx, batch);
      if (result != 0) {
        print('\n❌ 解码失败：$result');
        return;
      }

      nPast += nBatch;
    }

    // 生成循环
    const maxTokens = 256;
    final samplerChain = llama_sampler_chain_init(llama_sampler_chain_default_params());

    try {
      // 添加采样器：top-k, top-p (min_keep=1), temperature, dist
      llama_sampler_chain_add(samplerChain, llama_sampler_init_top_k(40));
      llama_sampler_chain_add(samplerChain, llama_sampler_init_top_p(0.9, 1));
      llama_sampler_chain_add(samplerChain, llama_sampler_init_temp(0.8));
      llama_sampler_chain_add(samplerChain, llama_sampler_init_dist(3407));

      for (var i = 0; i < maxTokens; i++) {
        // 获取下一个 token
        final nextToken = llama_sampler_sample(samplerChain, ctx, -1);

        // 检查是否结束
        if (nextToken == llama_token_eos(vocab)) {
          print('\n\n[结束]');
          break;
        }

        // 解码 token 为文本
        final tokenText = _decodeToken(vocab, nextToken);
        if (tokenText.isNotEmpty) {
          stdout.write(tokenText);
          stdout.flush();
        }

        // 准备下一次迭代
        batch.token[0] = nextToken;
        batch.pos[0] = nPast;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        batch.n_tokens = 1;

        // 解码
        final result = llama_decode(ctx, batch);
        if (result != 0) {
          print('\n❌ 解码失败：$result');
          break;
        }

        nPast++;

        // 模拟流式延迟
        await Future.delayed(const Duration(milliseconds: 50));
      }
    } finally {
      llama_sampler_free(samplerChain);
    }
  } finally {
    llama_batch_free(batch);
  }
}
