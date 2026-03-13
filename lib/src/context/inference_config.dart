import 'package:llama_native/src/sampling/sampling_config.dart';
import 'dart:io';

/// Llama 上下文推理配置
class InferenceConfig {
  /// 上下文长度
  final int nCtx;

  /// 批次大小
  final int nBatch;

  /// 线程数
  final int nThreads;

  /// 采样配置
  final SamplingConfig sampling;

  const InferenceConfig({
    this.nCtx = 4096,
    this.nBatch = 512,
    this.nThreads = 4,
    this.sampling = const SamplingConfig(),
  });

  /// 创建默认配置 (macOS 优化)
  factory InferenceConfig.defaultMacOS() {
    return InferenceConfig(
      nCtx: 4096,
      nBatch: 512,
      nThreads: Platform.numberOfProcessors,
      sampling: const SamplingConfig(temperature: 0.7, topP: 0.9),
    );
  }

  /// Android 默认配置
  factory InferenceConfig.defaultAndroid() {
    return InferenceConfig(
      nCtx: 2048,
      nBatch: 256,
      nThreads: Platform.numberOfProcessors.clamp(2, 4),
      sampling: const SamplingConfig(temperature: 0.7, topP: 0.9),
    );
  }

  /// Windows 默认配置
  factory InferenceConfig.defaultWindows() {
    return InferenceConfig(
      nCtx: 4096,
      nBatch: 512,
      nThreads: Platform.numberOfProcessors.clamp(2, 8),
      sampling: const SamplingConfig(temperature: 0.7, topP: 0.9),
    );
  }

  /// Linux 默认配置
  factory InferenceConfig.defaultLinux() {
    return InferenceConfig(
      nCtx: 4096,
      nBatch: 512,
      nThreads: Platform.numberOfProcessors.clamp(2, 8),
      sampling: const SamplingConfig(temperature: 0.7, topP: 0.9),
    );
  }

  /// iOS 默认配置
  factory InferenceConfig.defaultIOS() {
    return InferenceConfig(
      nCtx: 2048,
      nBatch: 256,
      nThreads: Platform.numberOfProcessors.clamp(2, 4),
      sampling: const SamplingConfig(temperature: 0.7, topP: 0.9),
    );
  }
}
