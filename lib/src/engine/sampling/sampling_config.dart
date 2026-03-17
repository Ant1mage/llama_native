import 'dart:typed_data';

/// Llama 基础采样配置
///
/// 负责：
/// - Temperature + Top-P 基础采样
/// - 代理到 llama.cpp 原生 sampler chain
/// - Repetition Penalty 基础支持
class SamplingConfig {
  /// 温度 (控制随机性)
  /// 0.0 = 确定性，1.0 = 原始分布，>1.0 = 更随机
  final double temperature;

  /// Top-P 核采样阈值
  /// 只保留累积概率前 P 的 token
  final double topP;

  /// Min-P 阈值
  /// 相对于最高概率 token 的最小比例
  final double minP;

  /// Top-K 采样
  /// 仅从 K 个最高概率 token 中采样
  final int topK;

  /// Repetition Penalty 重复惩罚
  /// 1.0 = 无惩罚，>1.0 = 惩罚重复
  final double penaltyRepeat;

  /// Penalty 应用的最后 N 个 token
  final int penaltyLastN;

  /// Frequency Penalty (频率惩罚)
  final double frequencyPenalty;

  /// Presence Penalty (存在惩罚)
  final double presencePenalty;

  const SamplingConfig({
    this.temperature = 0.7,
    this.topP = 0.9,
    this.minP = 0.0,
    this.topK = 40,
    this.penaltyRepeat = 1.1,
    this.penaltyLastN = 64,
    this.frequencyPenalty = 0.0,
    this.presencePenalty = 0.0,
  });

  /// 确定性采样 (greedy)
  const SamplingConfig.greedy()
    : temperature = 0.0,
      topP = 1.0,
      minP = 0.0,
      topK = 1,
      penaltyRepeat = 1.0,
      penaltyLastN = 0,
      frequencyPenalty = 0.0,
      presencePenalty = 0.0;

  /// 高创造性采样
  const SamplingConfig.creative()
    : temperature = 1.2,
      topP = 0.95,
      minP = 0.1,
      topK = 50,
      penaltyRepeat = 1.0,
      penaltyLastN = 64,
      frequencyPenalty = 0.3,
      presencePenalty = 0.3;

  /// 精确模式 (适合代码生成)
  const SamplingConfig.precise()
    : temperature = 0.2,
      topP = 0.9,
      minP = 0.0,
      topK = 20,
      penaltyRepeat = 1.0,
      penaltyLastN = 64,
      frequencyPenalty = 0.0,
      presencePenalty = 0.0;

  /// 转换为 native sampler chain params (简化实现)
  Map<String, dynamic> toNativeParams() {
    return {
      'temperature': temperature,
      'top_p': topP,
      'min_p': minP,
      'top_k': topK,
      'penalty_last_n': penaltyLastN,
      'penalty_repeat': penaltyRepeat,
      'freq_penalty': frequencyPenalty,
      'presence_penalty': presencePenalty,
    };
  }

  /// 创建采样器链 (简化实现)
  dynamic createSamplerChain() {
    // 实际实现应该调用 llama.cpp 的 API
    return null;
  }

  /// 合并配置
  SamplingConfig copyWith({
    double? temperature,
    double? topP,
    double? minP,
    int? topK,
    double? penaltyRepeat,
    int? penaltyLastN,
    double? frequencyPenalty,
    double? presencePenalty,
  }) {
    return SamplingConfig(
      temperature: temperature ?? this.temperature,
      topP: topP ?? this.topP,
      minP: minP ?? this.minP,
      topK: topK ?? this.topK,
      penaltyRepeat: penaltyRepeat ?? this.penaltyRepeat,
      penaltyLastN: penaltyLastN ?? this.penaltyLastN,
      frequencyPenalty: frequencyPenalty ?? this.frequencyPenalty,
      presencePenalty: presencePenalty ?? this.presencePenalty,
    );
  }

  @override
  String toString() {
    return 'LlamaSamplingConfig(temp=$temperature, topP=$topP, minP=$minP, topK=$topK)';
  }
}

/// Llama Logit Bias 配置 (干预特定词汇概率)
class LogitBias {
  /// token ID 到 bias 值的映射
  final Map<int, double> biases;

  /// 是否禁用某些 token
  final Set<int> disabledTokens;

  LogitBias({Map<int, double>? biases, Set<int>? disabledTokens})
    : biases = biases ?? {},
      disabledTokens = disabledTokens ?? {};

  /// 添加 bias
  void addBias(int token, double bias) {
    biases[token] = bias;
  }

  /// 禁用 token
  void disableToken(int token) {
    disabledTokens.add(token);
  }

  /// 应用 bias 到 logits
  void applyToLogits(Float32List logits) {
    // 应用 biases
    for (final entry in biases.entries) {
      if (entry.key >= 0 && entry.key < logits.length) {
        logits[entry.key] += entry.value;
      }
    }

    // 禁用 tokens (设置为负无穷)
    for (final token in disabledTokens) {
      if (token >= 0 && token < logits.length) {
        logits[token] = -1e9;
      }
    }
  }
}
