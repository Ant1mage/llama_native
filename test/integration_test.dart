import 'package:flutter_test/flutter_test.dart';
import 'package:llama_native/src/llama_native.dart';

void main() {
  group('ChatMessage', () {
    test('系统消息应该正确创建', () {
      const msg = ChatMessage.system('You are a helpful assistant.');
      expect(msg.role, equals(MessageRole.system));
      expect(msg.content, equals('You are a helpful assistant.'));
    });

    test('用户消息应该正确创建', () {
      const msg = ChatMessage.user('Hello!');
      expect(msg.role, equals(MessageRole.user));
      expect(msg.content, equals('Hello!'));
    });

    test('助手消息应该正确创建', () {
      const msg = ChatMessage.assistant('Hi there!');
      expect(msg.role, equals(MessageRole.assistant));
      expect(msg.content, equals('Hi there!'));
    });
  });

  group('ChatTemplateType', () {
    test('枚举值应该正确', () {
      expect(ChatTemplateType.llama3, isNotNull);
      expect(ChatTemplateType.qwen, isNotNull);
      expect(ChatTemplateType.mistral, isNotNull);
      expect(ChatTemplateType.chatml, isNotNull);
      expect(ChatTemplateType.alpaca, isNotNull);
    });
  });

  group('LlamaKVCacheManager', () {
    test('LlamaKVCacheManager 应该正确创建', () {
      final cache = KVCacheManager(nCtx: 4096);
      expect(cache.nPast, equals(0));
      expect(cache.nRemain, equals(4096));
    });
  });

  group('SessionState', () {
    test('会话 ID 应该自动生成', () {
      final session = SessionState();
      expect(session.sessionId, isNotEmpty);
      expect(session.sessionId.length, greaterThanOrEqualTo(16));
    });

    test('创建时间应该正确设置', () {
      final now = DateTime.now();
      final session = SessionState(createdAt: now);
      expect(session.createdAt, equals(now));
    });

    test('hasData 初始应为 false', () {
      final session = SessionState();
      expect(session.hasData, isFalse);
    });
  });

  group('SessionManager', () {
    test('创建会话应该返回新会话', () {
      final manager = SessionManager();
      final session = manager.createSession();

      expect(session, isNotNull);
      expect(manager.listSessions().length, equals(1));
    });

    test('获取会话应该返回正确的会话', () {
      final manager = SessionManager();
      final session = manager.createSession();

      final retrieved = manager.getSession(session.sessionId);
      expect(retrieved, equals(session));
    });

    test('删除会话应该移除会话', () {
      final manager = SessionManager();
      final session = manager.createSession();

      manager.removeSession(session.sessionId);
      expect(manager.listSessions().length, equals(0));
    });

    test('清空所有会话应该移除全部会话', () {
      final manager = SessionManager();
      manager.createSession();
      manager.createSession();

      manager.clearAll();
      expect(manager.listSessions().length, equals(0));
    });
  });

  group('LlamaContextInferenceConfig', () {
    test('默认配置应该有合理的值', () {
      const config = InferenceConfig();
      expect(config.nCtx, equals(4096));
      expect(config.nBatch, equals(512));
      expect(config.nThreads, equals(4));
    });

    test('采样配置应该正确传递', () {
      const sampling = SamplingConfig(temperature: 0.8);
      const config = InferenceConfig(sampling: sampling);
      expect(config.sampling.temperature, equals(0.8));
    });
  });
}
