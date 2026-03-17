import 'package:flutter_test/flutter_test.dart';
import 'package:llama_native/src/llama_native.dart';

void main() {
  group('LlamaModelConfig', () {
    test('配置应该正确保存参数', () {
      const config = LlamaModelConfig(modelPath: '/path/to/model.gguf', vocabOnly: true);
      expect(config.modelPath, equals('/path/to/model.gguf'));
      expect(config.vocabOnly, isTrue);
    });
  });

  group('LlamaModelMetadata', () {
    test('元数据应该正确保存', () {
      const metadata = LlamaModelMetadata(
        name: 'Test Model',
        parameterCount: 7.0,
        contextLength: 4096,
        embeddingDimension: 4096,
        layerCount: 32,
        vocabSize: 32000,
      );

      expect(metadata.name, equals('Test Model'));
      expect(metadata.parameterCount, equals(7.0));
      expect(metadata.contextLength, equals(4096));
      expect(metadata.embeddingDimension, equals(4096));
      expect(metadata.layerCount, equals(32));
      expect(metadata.vocabSize, equals(32000));
    });
  });

  group('LlamaSamplingConfig', () {
    test('默认采样配置应该有正确的值', () {
      const config = SamplingConfig();
      expect(config.temperature, closeTo(0.7, 0.01));
      expect(config.topP, closeTo(0.9, 0.01));
      expect(config.minP, equals(0.0));
      expect(config.topK, equals(40));
    });

    test('greedy 配置应该确定性采样', () {
      const config = SamplingConfig.greedy();
      expect(config.temperature, equals(0.0));
      expect(config.topK, equals(1));
    });

    test('creative 配置应该有高温度', () {
      const config = SamplingConfig.creative();
      expect(config.temperature, greaterThan(1.0));
      expect(config.topP, greaterThan(0.9));
    });

    test('precise 配置应该有低温度', () {
      const config = SamplingConfig.precise();
      expect(config.temperature, lessThan(0.3));
    });

    test('copyWith 应该创建新配置', () {
      const original = SamplingConfig(temperature: 0.5, topP: 0.8);
      final modified = original.copyWith(temperature: 1.0);

      expect(modified.temperature, equals(1.0));
      expect(modified.topP, equals(0.8)); // 保持不变
    });
  });

  group('LlamaLogitBias', () {
    test('bias 应该正确添加', () {
      final bias = LogitBias();
      bias.addBias(100, 5.0);
      bias.addBias(200, -3.0);

      expect(bias.biases.length, equals(2));
      expect(bias.biases[100], equals(5.0));
      expect(bias.biases[200], equals(-3.0));
    });

    test('禁用 token 应该正确记录', () {
      final bias = LogitBias();
      bias.disableToken(100);
      bias.disableToken(200);

      expect(bias.disabledTokens.contains(100), isTrue);
      expect(bias.disabledTokens.contains(200), isTrue);
    });
  });
}
