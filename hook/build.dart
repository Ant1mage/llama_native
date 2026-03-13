import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:native_toolchain_c/native_toolchain_c.dart';
import 'package:logging/logging.dart';

void main(List<String> args) async {
  await build(args, (input, output) async {
    final logger = Logger('')
      ..level = Level.ALL
      ..onRecord.listen((record) => print(record.message));

    if (input.config.buildCodeAssets) {
      // 检测目标平台
      final targetOS = input.config.code.targetOS;
      final targetArch = input.config.code.targetArchitecture.name;

      print('Building for $targetOS-$targetArch');

      // 根据平台选择 GPU 后端
      List<String> gpuSources = [];
      List<String> gpuIncludes = [];
      Map<String, String> gpuDefines = {};
      List<String> gpuFlags = [];

      switch (targetOS) {
        case OS.android:
        case OS.linux:
          // Linux/Android: CUDA 支持
          gpuSources = [
            'src/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu',
            // CUDA 核心操作 - 按需添加，这里列出主要的
            'src/llama.cpp/ggml/src/ggml-cuda/acc.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/add-id.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/arange.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/argmax.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/argsort.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/binbcast.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/clamp.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/concat.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/conv-transpose-1d.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/conv2d-dw.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/conv2d-transpose.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/conv2d.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/convert.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/count-equal.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/cpy.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/cross-entropy-loss.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/cumsum.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/diag.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/diagmask.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/fattn-tile.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/fattn-wmma-f16.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/fattn.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/fill.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/getrows.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/gla.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/im2col.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/mean.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/mmf.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/mmid.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/mmq.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/mmvf.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/mmvq.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/norm.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/opt-step-adamw.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/opt-step-sgd.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/out-prod.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/pad.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/pad_reflect_1d.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/pool2d.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/quantize.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/roll.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/rope.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/scale.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/set-rows.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/set.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/softcap.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/softmax.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/solve_tri.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/ssm-conv.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/ssm-scan.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/sum.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/sumrows.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/top-k.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/topk-moe.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/tri.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/tsembd.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/unary.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/upscale.cu',
            'src/llama.cpp/ggml/src/ggml-cuda/wkv.cu',
          ];
          gpuIncludes = ['src/llama.cpp/ggml/src/ggml-cuda'];
          gpuDefines = {
            'GGML_USE_CUDA': '1',
            'CUDA_ARCHITECTS': 'native', // 使用本地 GPU 架构
          };
          gpuFlags = [
            '--expendable-relaxed-constraints', // CUDA  relaxed 约束
          ];
          break;

        case OS.iOS:
        case OS.macOS:
          // macOS/iOS: Metal 支持
          gpuSources = [
            'src/llama.cpp/ggml/src/ggml-metal/ggml-metal.cpp',
            'src/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m',
            'src/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp',
            'src/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m',
            'src/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp',
          ];
          gpuIncludes = ['src/llama.cpp/ggml/src/ggml-metal'];
          gpuDefines = {
            'GGML_USE_METAL': '1',
            'GGML_METAL_NDEBUG': '1', // 禁用 Metal 调试
          };
          gpuFlags = [
            '-fobjc-arc', // 启用 Objective-C ARC
            '-framework', 'Metal',
            '-framework', 'Foundation',
            '-framework', 'QuartzCore', // 如果需要
          ];
          break;

        default:
          // Windows/Linux (无 GPU): 仅 CPU
          // Vulkan 支持需要额外配置，这里先使用 CPU
          break;
      }

      // llama.cpp 和 ggml 核心源文件
      final coreSources = [
        // llama.cpp 核心模块
        'src/llama.cpp/src/llama.cpp',
        'src/llama.cpp/src/llama-adapter.cpp',
        'src/llama.cpp/src/llama-arch.cpp',
        'src/llama.cpp/src/llama-batch.cpp',
        'src/llama.cpp/src/llama-chat.cpp',
        'src/llama.cpp/src/llama-context.cpp',
        'src/llama.cpp/src/llama-cparams.cpp',
        'src/llama.cpp/src/llama-grammar.cpp',
        'src/llama.cpp/src/llama-graph.cpp',
        'src/llama.cpp/src/llama-hparams.cpp',
        'src/llama.cpp/src/llama-impl.cpp',
        'src/llama.cpp/src/llama-io.cpp',
        'src/llama.cpp/src/llama-kv-cache.cpp',
        'src/llama.cpp/src/llama-kv-cache-iswa.cpp',
        'src/llama.cpp/src/llama-memory.cpp',
        'src/llama.cpp/src/llama-memory-hybrid.cpp',
        'src/llama.cpp/src/llama-memory-hybrid-iswa.cpp',
        'src/llama.cpp/src/llama-memory-recurrent.cpp',
        'src/llama.cpp/src/llama-mmap.cpp',
        'src/llama.cpp/src/llama-model.cpp',
        'src/llama.cpp/src/llama-model-loader.cpp',
        'src/llama.cpp/src/llama-model-saver.cpp',
        'src/llama.cpp/src/llama-quant.cpp', // 量化支持
        'src/llama.cpp/src/llama-sampler.cpp',
        'src/llama.cpp/src/llama-vocab.cpp',
        'src/llama.cpp/src/unicode.cpp',
        'src/llama.cpp/src/unicode-data.cpp',

        // ggml 核心模块
        'src/llama.cpp/ggml/src/ggml.c',
        'src/llama.cpp/ggml/src/ggml.cpp',
        'src/llama.cpp/ggml/src/ggml-alloc.c',
        'src/llama.cpp/ggml/src/ggml-backend.cpp',
        'src/llama.cpp/ggml/src/ggml-backend-reg.cpp',
        'src/llama.cpp/ggml/src/ggml-opt.cpp',
        'src/llama.cpp/ggml/src/ggml-quants.c', // 量化核心
        'src/llama.cpp/ggml/src/ggml-threading.cpp',
        'src/llama.cpp/ggml/src/gguf.cpp',

        // CPU backend (必需，作为后备)
        'src/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c',
        'src/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp',
        'src/llama.cpp/ggml/src/ggml-cpu/quants.c',
        'src/llama.cpp/ggml/src/ggml-cpu/hbm.cpp',
        'src/llama.cpp/ggml/src/ggml-cpu/traits.cpp',
      ];

      // 合并所有源文件
      final allSources = [...coreSources, ...gpuSources];

      // 构建 llama.cpp 库
      final cBuilder = CBuilder.library(
        name: 'llama_native',
        assetName: 'llama_native.dart',
        sources: allSources,
        includes: [
          'src/llama.cpp/include',
          'src/llama.cpp/ggml/include',
          'src/llama.cpp/src',
          'src/llama.cpp/common',
          'src/llama.cpp/ggml/src',
          'src/llama.cpp/ggml/src/ggml-cpu',
          ...gpuIncludes,
        ],
        defines: {
          // ========== 量化支持 ==========
          'GGML_USE_CPU': '1', // CPU 作为后备
          'GGML_SCHED_MAX_COPIES': '4',

          // 量化类型支持
          'K_QUANTS_PER_ITERATION': '2',
          'GGML_DEFAULT_N_THREADS': '4',

          // 优化选项
          'GGML_USE_OPENMP': '1', // OpenMP 并行
          'GGML_USE_THREADING': '1',

          // 内存优化
          'GGML_USE_HBM': '1', // 高带宽内存
          // 合并到 GPU defines
          ...gpuDefines,
        },
        flags: [
          // ========== 优化标志 ==========
          // 通用优化
          '-O3', // 最高级别优化
          '-ffast-math', // 快速数学运算（可能损失精度）
          '-funroll-loops', // 循环展开
          '-fomit-frame-pointer', // 省略帧指针
          '-fstrict-aliasing', // 严格别名
          '-ftree-vectorize', // 自动向量化
          // ARM 架构优化
          if (targetArch == 'arm64') ...[
            '-mcpu=native', // 针对本地 CPU 优化
            '-mtune=native',
          ],

          // 针对 x86_64 架构
          if (targetArch == 'x86_64') ...[
            '-march=native', // 针对本地 CPU 优化
            '-mtune=native',
            '-mavx2', // AVX2 指令集
            '-mfma', // FMA 指令集
            '-mf16c', // FP16 转换指令
          ],

          // 平台特定标志
          if (targetOS == OS.android) ...[
            '-fPIC', // 位置无关代码
          ],

          if (targetOS == OS.iOS || targetOS == OS.macOS) ...[
            '-fPIC',
            '-fembed-bitcode', // 嵌入 bitcode（iOS 需要）
          ],

          // 合并 GPU flags
          ...gpuFlags,
        ],
      );

      await cBuilder.run(input: input, output: output, logger: logger);
    }
  });
}
