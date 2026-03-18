class PerformanceMetrics {
  final double tStartMs;
  final double tLoadMs;
  final double tPromptEvalMs;
  final double tEvalMs;
  final int nPromptEval;
  final int nEval;
  final int nReused;
  final double tSampleMs;
  final int nSample;

  const PerformanceMetrics({
    this.tStartMs = 0,
    this.tLoadMs = 0,
    this.tPromptEvalMs = 0,
    this.tEvalMs = 0,
    this.nPromptEval = 0,
    this.nEval = 0,
    this.nReused = 0,
    this.tSampleMs = 0,
    this.nSample = 0,
  });

  factory PerformanceMetrics.empty() => const PerformanceMetrics();

  double get totalEvalMs => tPromptEvalMs + tEvalMs + tSampleMs;

  int get totalTokens => nPromptEval + nEval;

  double get msPerToken {
    if (nEval == 0) return 0;
    return tEvalMs / nEval;
  }

  double get msPerPromptToken {
    if (nPromptEval == 0) return 0;
    return tPromptEvalMs / nPromptEval;
  }

  double get tokensPerSecond {
    if (tEvalMs == 0) return 0;
    return nEval / (tEvalMs / 1000);
  }

  double get promptTokensPerSecond {
    if (tPromptEvalMs == 0) return 0;
    return nPromptEval / (tPromptEvalMs / 1000);
  }

  bool get isOverloaded => msPerToken > 200;

  bool get isSlowDevice => tokensPerSecond < 5;

  @override
  String toString() {
    return 'PerformanceMetrics('
        'prompt: ${nPromptEval}t @ ${promptTokensPerSecond.toStringAsFixed(1)} t/s, '
        'eval: ${nEval}t @ ${tokensPerSecond.toStringAsFixed(1)} t/s, '
        'ms/tok: ${msPerToken.toStringAsFixed(1)}'
        ')';
  }

  Map<String, dynamic> toMap() {
    return {
      'tStartMs': tStartMs,
      'tLoadMs': tLoadMs,
      'tPromptEvalMs': tPromptEvalMs,
      'tEvalMs': tEvalMs,
      'nPromptEval': nPromptEval,
      'nEval': nEval,
      'nReused': nReused,
      'tSampleMs': tSampleMs,
      'nSample': nSample,
      'msPerToken': msPerToken,
      'tokensPerSecond': tokensPerSecond,
      'isOverloaded': isOverloaded,
    };
  }
}
