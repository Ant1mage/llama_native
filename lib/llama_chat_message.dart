/// Chat 消息角色
enum LlamaChatMessageRole { system, user, assistant }

/// Chat 消息
class LlamaChatMessage {
  final LlamaChatMessageRole role;
  final String content;

  const LlamaChatMessage({required this.role, required this.content});

  const LlamaChatMessage.system(this.content) : role = LlamaChatMessageRole.system;
  const LlamaChatMessage.user(this.content) : role = LlamaChatMessageRole.user;
  const LlamaChatMessage.assistant(this.content) : role = LlamaChatMessageRole.assistant;

  Map<String, String> toMap() => {
    'role': switch (role) {
      LlamaChatMessageRole.system => 'system',
      LlamaChatMessageRole.user => 'user',
      LlamaChatMessageRole.assistant => 'assistant',
    },
    'content': content,
  };
}
