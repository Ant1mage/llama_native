import 'dart:typed_data';

enum SamplerType { none, dry, topK, topP, minP, typicalP, temperature, xtc, infill, penalties, topNSigma, adaptiveP }

class SamplingConfig {
  final double temperature;
  final double topP;
  final double minP;
  final int topK;
  final double penaltyRepeat;
  final int penaltyLastN;
  final double frequencyPenalty;
  final double presencePenalty;

  final double typP;
  final double topNSigma;
  final double xtcProbability;
  final double xtcThreshold;
  final int minKeep;

  final int mirostat;
  final double mirostatTau;
  final double mirostatEta;

  final double dynatempRange;
  final double dynatempExponent;

  final double dryMultiplier;
  final double dryBase;
  final int dryAllowedLength;
  final int dryPenaltyLastN;
  final List<String> drySequenceBreakers;

  final double adaptiveTarget;
  final double adaptiveDecay;

  final int seed;
  final bool ignoreEos;
  final List<SamplerType> samplers;

  const SamplingConfig({
    this.temperature = 0.8,
    this.topP = 0.95,
    this.minP = 0.05,
    this.topK = 40,
    this.penaltyRepeat = 1.0,
    this.penaltyLastN = 64,
    this.frequencyPenalty = 0.0,
    this.presencePenalty = 0.0,
    this.typP = 1.0,
    this.topNSigma = -1.0,
    this.xtcProbability = 0.0,
    this.xtcThreshold = 0.1,
    this.minKeep = 1,
    this.mirostat = 0,
    this.mirostatTau = 5.0,
    this.mirostatEta = 0.1,
    this.dynatempRange = 0.0,
    this.dynatempExponent = 1.0,
    this.dryMultiplier = 0.0,
    this.dryBase = 1.75,
    this.dryAllowedLength = 2,
    this.dryPenaltyLastN = -1,
    this.drySequenceBreakers = const ['\n', ':', '"', '*'],
    this.adaptiveTarget = -1.0,
    this.adaptiveDecay = 0.9,
    this.seed = 0xFFFFFFFF,
    this.ignoreEos = false,
    this.samplers = const [
      SamplerType.penalties,
      SamplerType.dry,
      SamplerType.topNSigma,
      SamplerType.topK,
      SamplerType.typicalP,
      SamplerType.topP,
      SamplerType.minP,
      SamplerType.xtc,
      SamplerType.temperature,
    ],
  });

  const SamplingConfig.greedy()
    : temperature = 0.0,
      topP = 1.0,
      minP = 0.0,
      topK = 1,
      penaltyRepeat = 1.0,
      penaltyLastN = 0,
      frequencyPenalty = 0.0,
      presencePenalty = 0.0,
      typP = 1.0,
      topNSigma = -1.0,
      xtcProbability = 0.0,
      xtcThreshold = 0.1,
      minKeep = 1,
      mirostat = 0,
      mirostatTau = 5.0,
      mirostatEta = 0.1,
      dynatempRange = 0.0,
      dynatempExponent = 1.0,
      dryMultiplier = 0.0,
      dryBase = 1.75,
      dryAllowedLength = 2,
      dryPenaltyLastN = -1,
      drySequenceBreakers = const ['\n', ':', '"', '*'],
      adaptiveTarget = -1.0,
      adaptiveDecay = 0.9,
      seed = 0xFFFFFFFF,
      ignoreEos = false,
      samplers = const [SamplerType.temperature];

  const SamplingConfig.creative()
    : temperature = 1.2,
      topP = 0.95,
      minP = 0.1,
      topK = 50,
      penaltyRepeat = 1.0,
      penaltyLastN = 64,
      frequencyPenalty = 0.3,
      presencePenalty = 0.3,
      typP = 1.0,
      topNSigma = -1.0,
      xtcProbability = 0.0,
      xtcThreshold = 0.1,
      minKeep = 1,
      mirostat = 0,
      mirostatTau = 5.0,
      mirostatEta = 0.1,
      dynatempRange = 0.0,
      dynatempExponent = 1.0,
      dryMultiplier = 0.0,
      dryBase = 1.75,
      dryAllowedLength = 2,
      dryPenaltyLastN = -1,
      drySequenceBreakers = const ['\n', ':', '"', '*'],
      adaptiveTarget = -1.0,
      adaptiveDecay = 0.9,
      seed = 0xFFFFFFFF,
      ignoreEos = false,
      samplers = const [
        SamplerType.penalties,
        SamplerType.dry,
        SamplerType.topNSigma,
        SamplerType.topK,
        SamplerType.typicalP,
        SamplerType.topP,
        SamplerType.minP,
        SamplerType.xtc,
        SamplerType.temperature,
      ];

  const SamplingConfig.precise()
    : temperature = 0.2,
      topP = 0.9,
      minP = 0.0,
      topK = 20,
      penaltyRepeat = 1.0,
      penaltyLastN = 64,
      frequencyPenalty = 0.0,
      presencePenalty = 0.0,
      typP = 1.0,
      topNSigma = -1.0,
      xtcProbability = 0.0,
      xtcThreshold = 0.1,
      minKeep = 1,
      mirostat = 0,
      mirostatTau = 5.0,
      mirostatEta = 0.1,
      dynatempRange = 0.0,
      dynatempExponent = 1.0,
      dryMultiplier = 0.0,
      dryBase = 1.75,
      dryAllowedLength = 2,
      dryPenaltyLastN = -1,
      drySequenceBreakers = const ['\n', ':', '"', '*'],
      adaptiveTarget = -1.0,
      adaptiveDecay = 0.9,
      seed = 0xFFFFFFFF,
      ignoreEos = false,
      samplers = const [SamplerType.penalties, SamplerType.topK, SamplerType.topP, SamplerType.temperature];

  const SamplingConfig.mirostat({double tau = 5.0, double eta = 0.1, this.mirostat = 2})
    : temperature = 0.8,
      topP = 1.0,
      minP = 0.0,
      topK = 40,
      penaltyRepeat = 1.0,
      penaltyLastN = 64,
      frequencyPenalty = 0.0,
      presencePenalty = 0.0,
      typP = 1.0,
      topNSigma = -1.0,
      xtcProbability = 0.0,
      xtcThreshold = 0.1,
      minKeep = 1,
      mirostatTau = tau,
      mirostatEta = eta,
      dynatempRange = 0.0,
      dynatempExponent = 1.0,
      dryMultiplier = 0.0,
      dryBase = 1.75,
      dryAllowedLength = 2,
      dryPenaltyLastN = -1,
      drySequenceBreakers = const ['\n', ':', '"', '*'],
      adaptiveTarget = -1.0,
      adaptiveDecay = 0.9,
      seed = 0xFFFFFFFF,
      ignoreEos = false,
      samplers = const [SamplerType.temperature];

  bool get hasPenalties => penaltyRepeat != 1.0 || frequencyPenalty != 0.0 || presencePenalty != 0.0;

  bool get hasDry => dryMultiplier > 0.0;

  bool get hasTopNSigma => topNSigma > 0.0;

  bool get hasTypicalP => typP < 1.0;

  bool get hasTopP => topP < 1.0;

  bool get hasMinP => minP > 0.0;

  bool get hasXtc => xtcProbability > 0.0 && xtcThreshold <= 0.5;

  bool get hasAdaptiveP => adaptiveTarget >= 0.0 && adaptiveTarget <= 1.0;

  bool get isGreedy => temperature <= 0.0;

  bool get useMirostat => mirostat > 0;

  bool get hasDynatemp => dynatempRange > 0.0;

  SamplingConfig copyWith({
    double? temperature,
    double? topP,
    double? minP,
    int? topK,
    double? penaltyRepeat,
    int? penaltyLastN,
    double? frequencyPenalty,
    double? presencePenalty,
    double? typP,
    double? topNSigma,
    double? xtcProbability,
    double? xtcThreshold,
    int? minKeep,
    int? mirostat,
    double? mirostatTau,
    double? mirostatEta,
    double? dynatempRange,
    double? dynatempExponent,
    double? dryMultiplier,
    double? dryBase,
    int? dryAllowedLength,
    int? dryPenaltyLastN,
    List<String>? drySequenceBreakers,
    double? adaptiveTarget,
    double? adaptiveDecay,
    int? seed,
    bool? ignoreEos,
    List<SamplerType>? samplers,
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
      typP: typP ?? this.typP,
      topNSigma: topNSigma ?? this.topNSigma,
      xtcProbability: xtcProbability ?? this.xtcProbability,
      xtcThreshold: xtcThreshold ?? this.xtcThreshold,
      minKeep: minKeep ?? this.minKeep,
      mirostat: mirostat ?? this.mirostat,
      mirostatTau: mirostatTau ?? this.mirostatTau,
      mirostatEta: mirostatEta ?? this.mirostatEta,
      dynatempRange: dynatempRange ?? this.dynatempRange,
      dynatempExponent: dynatempExponent ?? this.dynatempExponent,
      dryMultiplier: dryMultiplier ?? this.dryMultiplier,
      dryBase: dryBase ?? this.dryBase,
      dryAllowedLength: dryAllowedLength ?? this.dryAllowedLength,
      dryPenaltyLastN: dryPenaltyLastN ?? this.dryPenaltyLastN,
      drySequenceBreakers: drySequenceBreakers ?? this.drySequenceBreakers,
      adaptiveTarget: adaptiveTarget ?? this.adaptiveTarget,
      adaptiveDecay: adaptiveDecay ?? this.adaptiveDecay,
      seed: seed ?? this.seed,
      ignoreEos: ignoreEos ?? this.ignoreEos,
      samplers: samplers ?? this.samplers,
    );
  }

  @override
  String toString() {
    return 'SamplingConfig(temp=$temperature, topP=$topP, minP=$minP, topK=$topK, mirostat=$mirostat)';
  }
}

class LogitBias {
  final Map<int, double> biases;
  final Set<int> disabledTokens;

  LogitBias({Map<int, double>? biases, Set<int>? disabledTokens})
    : biases = biases ?? {},
      disabledTokens = disabledTokens ?? {};

  void addBias(int token, double bias) {
    biases[token] = bias;
  }

  void disableToken(int token) {
    disabledTokens.add(token);
  }

  bool get hasBiases => biases.isNotEmpty || disabledTokens.isNotEmpty;

  void applyToLogits(Float32List logits) {
    for (final entry in biases.entries) {
      if (entry.key >= 0 && entry.key < logits.length) {
        logits[entry.key] += entry.value;
      }
    }

    for (final token in disabledTokens) {
      if (token >= 0 && token < logits.length) {
        logits[token] = -1e9;
      }
    }
  }
}
