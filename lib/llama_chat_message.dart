/// 消息角色
enum LlamaMessageRole { system, user, assistant }

/// 消息
class LlamaChatMessage {
  final LlamaMessageRole role;
  final String content;

  const LlamaChatMessage({required this.role, required this.content});

  const LlamaChatMessage.system(this.content) : role = LlamaMessageRole.system;
  const LlamaChatMessage.user(this.content) : role = LlamaMessageRole.user;
  const LlamaChatMessage.assistant(this.content) : role = LlamaMessageRole.assistant;

  Map<String, String> toMap() => {
    'role': switch (role) {
      LlamaMessageRole.system => 'system',
      LlamaMessageRole.user => 'user',
      LlamaMessageRole.assistant => 'assistant',
    },
    'content': content,
  };
}
