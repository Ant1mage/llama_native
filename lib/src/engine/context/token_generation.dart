class TokenGeneration {
  final int token;
  final String text;
  final bool isEnd;

  final int kvUsed;
  final int kvTotal;
  final double kvUsagePercent;

  const TokenGeneration({
    required this.token,
    required this.text,
    this.isEnd = false,
    this.kvUsed = 0,
    this.kvTotal = 0,
    this.kvUsagePercent = 0.0,
  });

  TokenGeneration copyWith({
    int? token,
    String? text,
    bool? isEnd,
    int? kvUsed,
    int? kvTotal,
    double? kvUsagePercent,
  }) {
    return TokenGeneration(
      token: token ?? this.token,
      text: text ?? this.text,
      isEnd: isEnd ?? this.isEnd,
      kvUsed: kvUsed ?? this.kvUsed,
      kvTotal: kvTotal ?? this.kvTotal,
      kvUsagePercent: kvUsagePercent ?? this.kvUsagePercent,
    );
  }
}

class KVCacheStatus {
  final int used;
  final int total;
  final int safeCapacity;
  final int moeBufferSlots;
  final double usagePercent;
  final bool isOverSafeThreshold;
  final bool isFull;

  const KVCacheStatus({
    required this.used,
    required this.total,
    required this.safeCapacity,
    required this.moeBufferSlots,
    required this.usagePercent,
    required this.isOverSafeThreshold,
    required this.isFull,
  });

  factory KVCacheStatus.empty() {
    return const KVCacheStatus(
      used: 0,
      total: 0,
      safeCapacity: 0,
      moeBufferSlots: 0,
      usagePercent: 0.0,
      isOverSafeThreshold: false,
      isFull: false,
    );
  }

  bool get needsReset => isOverSafeThreshold;
  int get remaining => total - used;
}
