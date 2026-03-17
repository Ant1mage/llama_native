import 'package:llama_native/src/engine/sampling/sampling_config.dart';
import 'package:llama_native/src/utils/platform_info.dart';
import 'dart:io';

class InferenceConfig {
  final int nCtx;
  final int nBatch;
  final int nUBatch;
  final int nThreads;
  final int nGpuLayers;
  final int nGpuLayersDraft;
  final SamplingConfig sampling;

  const InferenceConfig({
    this.nCtx = 4096,
    this.nBatch = 512,
    this.nUBatch = 128,
    this.nThreads = 4,
    this.nGpuLayers = 0,
    this.nGpuLayersDraft = 0,
    this.sampling = const SamplingConfig(),
  });

  factory InferenceConfig.defaults({
    int? nCtx,
    int? nBatch,
    int? nUBatch,
    int? nThreads,
    int? nGpuLayers,
    SamplingConfig? sampling,
    int? modelSizeMB,
    int? modelLayers,
  }) {
    return InferenceConfig(
      nCtx: nCtx ?? PlatformInfo.recommendedContextLength(modelSizeMB: modelSizeMB),
      nBatch: nBatch ?? PlatformInfo.recommendedBatchSize(),
      nUBatch: nUBatch ?? PlatformInfo.recommendedUBatchSize(),
      nThreads: nThreads ?? PlatformInfo.recommendedThreads(),
      nGpuLayers: nGpuLayers ?? PlatformInfo.recommendedGpuLayers(modelSizeMB: modelSizeMB, modelLayers: modelLayers),
      sampling: sampling ?? const SamplingConfig(temperature: 0.7, topP: 0.9),
    );
  }

  factory InferenceConfig.forModel({
    required int modelSizeMB,
    int? modelLayers,
    int? nCtx,
    int? nThreads,
    SamplingConfig? sampling,
  }) {
    return InferenceConfig.defaults(
      modelSizeMB: modelSizeMB,
      modelLayers: modelLayers,
      nCtx: nCtx,
      nThreads: nThreads,
      sampling: sampling,
    );
  }

  factory InferenceConfig.defaultMacOS({
    int? nCtx,
    int? nBatch,
    int? nUBatch,
    int? nThreads,
    int? nGpuLayers,
    SamplingConfig? sampling,
  }) {
    return InferenceConfig(
      nCtx: nCtx ?? 8192,
      nBatch: nBatch ?? 512,
      nUBatch: nUBatch ?? 128,
      nThreads: nThreads ?? PlatformInfo.recommendedThreads(),
      nGpuLayers: nGpuLayers ?? (PlatformInfo.supportsGpuOffload ? 99 : 0),
      sampling: sampling ?? const SamplingConfig(temperature: 0.7, topP: 0.9),
    );
  }

  factory InferenceConfig.defaultIOS({
    int? nCtx,
    int? nBatch,
    int? nUBatch,
    int? nThreads,
    int? nGpuLayers,
    SamplingConfig? sampling,
  }) {
    return InferenceConfig(
      nCtx: nCtx ?? 2048,
      nBatch: nBatch ?? 64,
      nUBatch: nUBatch ?? 32,
      nThreads: nThreads ?? Platform.numberOfProcessors.clamp(2, 4),
      nGpuLayers: nGpuLayers ?? (PlatformInfo.supportsGpuOffload ? 40 : 0),
      sampling: sampling ?? const SamplingConfig(temperature: 0.7, topP: 0.9),
    );
  }

  factory InferenceConfig.defaultAndroid({
    int? nCtx,
    int? nBatch,
    int? nUBatch,
    int? nThreads,
    int? nGpuLayers,
    SamplingConfig? sampling,
  }) {
    return InferenceConfig(
      nCtx: nCtx ?? 2048,
      nBatch: nBatch ?? 256,
      nUBatch: nUBatch ?? 128,
      nThreads: nThreads ?? Platform.numberOfProcessors.clamp(2, 4),
      nGpuLayers: nGpuLayers ?? (PlatformInfo.supportsGpuOffload ? 20 : 0),
      sampling: sampling ?? const SamplingConfig(temperature: 0.7, topP: 0.9),
    );
  }

  factory InferenceConfig.defaultWindows({
    int? nCtx,
    int? nBatch,
    int? nUBatch,
    int? nThreads,
    int? nGpuLayers,
    SamplingConfig? sampling,
  }) {
    return InferenceConfig(
      nCtx: nCtx ?? 4096,
      nBatch: nBatch ?? 512,
      nUBatch: nUBatch ?? 128,
      nThreads: nThreads ?? Platform.numberOfProcessors.clamp(2, 8),
      nGpuLayers: nGpuLayers ?? (PlatformInfo.supportsGpuOffload ? 35 : 0),
      sampling: sampling ?? const SamplingConfig(temperature: 0.7, topP: 0.9),
    );
  }

  factory InferenceConfig.defaultLinux({
    int? nCtx,
    int? nBatch,
    int? nUBatch,
    int? nThreads,
    int? nGpuLayers,
    SamplingConfig? sampling,
  }) {
    return InferenceConfig(
      nCtx: nCtx ?? 4096,
      nBatch: nBatch ?? 512,
      nUBatch: nUBatch ?? 128,
      nThreads: nThreads ?? Platform.numberOfProcessors.clamp(2, 8),
      nGpuLayers: nGpuLayers ?? (PlatformInfo.supportsGpuOffload ? 35 : 0),
      sampling: sampling ?? const SamplingConfig(temperature: 0.7, topP: 0.9),
    );
  }

  factory InferenceConfig.cpu({int? nCtx, int? nBatch, int? nThreads, SamplingConfig? sampling}) {
    return InferenceConfig(
      nCtx: nCtx ?? 4096,
      nBatch: nBatch ?? 256,
      nUBatch: 128,
      nThreads: nThreads ?? PlatformInfo.recommendedThreads(),
      nGpuLayers: 0,
      sampling: sampling ?? const SamplingConfig(temperature: 0.7, topP: 0.9),
    );
  }

  factory InferenceConfig.gpu({
    int? nCtx,
    int? nBatch,
    int? nGpuLayers,
    SamplingConfig? sampling,
    int? modelSizeMB,
    int? modelLayers,
  }) {
    return InferenceConfig(
      nCtx: nCtx ?? PlatformInfo.recommendedContextLength(modelSizeMB: modelSizeMB),
      nBatch: nBatch ?? PlatformInfo.recommendedBatchSize(useGpu: true),
      nUBatch: 128,
      nThreads: PlatformInfo.recommendedThreads(),
      nGpuLayers: nGpuLayers ?? PlatformInfo.recommendedGpuLayers(modelSizeMB: modelSizeMB, modelLayers: modelLayers),
      sampling: sampling ?? const SamplingConfig(temperature: 0.7, topP: 0.9),
    );
  }

  InferenceConfig copyWith({
    int? nCtx,
    int? nBatch,
    int? nUBatch,
    int? nThreads,
    int? nGpuLayers,
    int? nGpuLayersDraft,
    SamplingConfig? sampling,
  }) {
    return InferenceConfig(
      nCtx: nCtx ?? this.nCtx,
      nBatch: nBatch ?? this.nBatch,
      nUBatch: nUBatch ?? this.nUBatch,
      nThreads: nThreads ?? this.nThreads,
      nGpuLayers: nGpuLayers ?? this.nGpuLayers,
      nGpuLayersDraft: nGpuLayersDraft ?? this.nGpuLayersDraft,
      sampling: sampling ?? this.sampling,
    );
  }

  @override
  String toString() {
    return 'InferenceConfig(nCtx: $nCtx, nBatch: $nBatch, nUBatch: $nUBatch, '
        'nThreads: $nThreads, nGpuLayers: $nGpuLayers, sampling: $sampling)';
  }
}
