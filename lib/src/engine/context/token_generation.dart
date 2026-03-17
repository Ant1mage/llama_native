/// Llama Token 生成结果
class TokenGeneration {
  /// 生成的 token ID
  final int token;

  /// 解码后的文本
  final String text;

  /// 是否结束 token
  final bool isEnd;

  const TokenGeneration({required this.token, required this.text, this.isEnd = false});
}
