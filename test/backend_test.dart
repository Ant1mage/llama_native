import 'package:flutter_test/flutter_test.dart';
import 'package:llama_native/llama_native.dart';

void main() {
  group('LlamaBackend', () {
    test('单例模式应该返回相同实例', () {
      final backend1 = LlamaBackend.instance;
      final backend2 = LlamaBackend.instance;
      expect(backend1, equals(backend2));
    });

    test('platform 检测应该返回正确平台', () {
      final backend = LlamaBackend.instance;
      expect(backend.currentPlatform, isNotEmpty);
    });

    test('硬件加速检测应该返回有效值', () {
      final backend = LlamaBackend.instance;
      final acceleration = backend.detectHardwareAcceleration();
      expect(acceleration, isNotNull);
    });

    test('isInitialized 初始应为 false', () {
      final backend = LlamaBackend.instance;
      expect(backend.isInitialized, isFalse);
    });
  });

  group('LlamaBackendConfig', () {
    test('默认配置应该有正确的默认值', () {
      const config = LlamaBackendConfig();
      expect(config.gpuLayers, equals(0));
      expect(config.useMmap, isTrue);
      expect(config.useMlock, isFalse);
    });

    test('自定义配置应该正确设置参数', () {
      const config = LlamaBackendConfig(gpuLayers: 50, useMmap: false, useMlock: true);
      expect(config.gpuLayers, equals(50));
      expect(config.useMmap, isFalse);
      expect(config.useMlock, isTrue);
    });
  });

  group('LlamaPlatformInfo', () {
    test('recommendedGpuLayers 应该返回非负值', () {
      final layers = PlatformInfo.recommendedGpuLayers();
      expect(layers, greaterThanOrEqualTo(0));
    });

    test('recommendedContextLength 应该返回合理值', () {
      final ctxLength = PlatformInfo.recommendedContextLength();
      expect(ctxLength, greaterThanOrEqualTo(1024));
    });
  });
}
