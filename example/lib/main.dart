import 'dart:async';
import 'dart:io';

import 'package:llama_native/llama_native.dart';

void main(List<String> args) async {
  print('🦙 LLaMA Native - 流式对话示例 (高级 API)');
  print('=' * 50);

  // 1. 初始化后端
  print('\n📦 初始化 llama 后端...');
  final backend = LlamaBackend.instance;
  await backend.initialize();

  try {
    // 2. 加载模型
    final modelPath = '../models/Qwen3.5-4B-Q4_K_M.gguf';
    print('\n📂 加载模型：$modelPath');

    if (!File(modelPath).existsSync()) {
      print('❌ 模型文件不存在：$modelPath');
      exit(1);
    }

    final model = LlamaModel.load(LlamaModelConfig(modelPath: modelPath, vocabOnly: false));

    print('✅ 模型加载成功');

    // 打印模型信息
    final metadata = model.getMetadata();
    print('   名称：${metadata.name}');
    print('   参数：${metadata.parameterCount.toStringAsFixed(1)}B');
    print('   上下文：${metadata.contextLength}');
    print('   词表：${metadata.vocabSize}');

    // 3. 创建推理配置
    final inferenceConfig = InferenceConfig.defaultMacOS();

    // 4. 创建上下文
    print('\n🔧 创建推理上下文...');
    final context = LlamaContext.create(model, inferenceConfig);
    print('✅ 上下文创建成功');

    // 5. 初始化 ChatTokenizer
    final tokenizer = ChatTokenizer(
      model: model,
      templateType: ChatTemplateType.qwen, // 自动检测也可以
      addBos: true,
      addEos: false,
    );

    // 6. 构建对话历史
    final messages = <ChatMessage>[const ChatMessage.system('你是一个有帮助的 AI 助手。'), const ChatMessage.user('你是谁？')];

    print('\n💬 用户：${messages.last.content}');

    // 7. Tokenize 对话
    final inputTokens = tokenizer.tokenizeMessages(messages);
    print('   Tokens: ${inputTokens.length}');

    // 8. 流式生成
    print('\n🤖 助手：');
    await _generateStreamed(context, inputTokens);

    // 9. 清理资源
    print('\n\n🧹 清理资源...');
    context.dispose();
    model.dispose();
    print('✅ 资源清理完成');
  } catch (e, stackTrace) {
    print('❌ 错误：$e');
    print(stackTrace);
    backend.dispose();
    exit(1);
  }

  // 10. 释放后端
  print('\n👋 释放后端...');
  backend.dispose();
  print('✅ 程序结束');
}

/// 流式生成
Future<void> _generateStreamed(LlamaContext context, List<int> inputTokens) async {
  final maxTokens = 256;
  var generatedCount = 0;

  // 使用流式生成
  final stream = context.generateStream(inputTokens, maxTokens: maxTokens);

  await for (final result in stream) {
    if (result.text.isNotEmpty) {
      stdout.write(result.text);
      stdout.flush();
      generatedCount++;
    }

    if (result.isEnd) {
      print('\n[EOS]');
      break;
    }

    // 模拟流式延迟
    await Future.delayed(const Duration(milliseconds: 50));
  }

  print('\n生成了 $generatedCount 个 tokens');
}
