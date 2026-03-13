import 'dart:io';
import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:path/path.dart' as p;

void main(List<String> args) async {
  await build(args, (input, output) async {
    final targetOS = input.config.code.targetOS;
    final targetArch = input.config.code.targetArchitecture;
    final packageRoot = input.packageRoot.toFilePath();

    // 1. 设置构建目录 (放在 .dart_tool 目录下以保持工程整洁)
    final buildDir = Directory(p.join(input.outputDirectory.toFilePath(), 'cmake_build'));
    if (!buildDir.existsSync()) buildDir.createSync(recursive: true);

    print('🚀 Preparing [llama_native] for $targetOS-${targetArch.name}');

    // 2. 配置 CMake 参数 - 使用 llama.cpp 自带的 CMakeLists.txt
    final cmakeArgs = <String>[
      '-S', p.join(packageRoot, 'src', 'llama.cpp'),
      '-B', buildDir.path,
      '-DCMAKE_BUILD_TYPE=Release',
      '-DBUILD_SHARED_LIBS=ON',
      // 禁用不必要的组件以加速构建
      '-DLLAMA_BUILD_TESTS=OFF',
      '-DLLAMA_BUILD_EXAMPLES=OFF',
      '-DLLAMA_BUILD_TOOLS=OFF',
      '-DLLAMA_BUILD_SERVER=OFF',
      // 启用核心库和通用工具
      '-DLLAMA_BUILD_COMMON=ON',
      // 禁用原生 CPU 优化（避免 macOS 上的 -mcpu=native 错误）
      '-DGGML_NATIVE=OFF',
      // macOS RPATH 配置：确保运行时能找到依赖库
      '-DCMAKE_INSTALL_RPATH=@executable_path/../Frameworks;@loader_path/../Frameworks',
      '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON',
      '-DCMAKE_INSTALL_NAME_DIR=@rpath',
    ];

    // 分平台配置
    if (targetOS == OS.android) {
      // Android NDK 配置
      final androidNdkHome = Platform.environment['ANDROID_NDK_HOME'];
      if (androidNdkHome != null) {
        cmakeArgs.addAll([
          '-DCMAKE_TOOLCHAIN_FILE=${p.join(androidNdkHome, 'build/cmake/android.toolchain.cmake')}',
          '-DANDROID_ABI=${_mapAndroidAbi(targetArch)}',
          '-DANDROID_PLATFORM=android-24',
        ]);
      }
    } else if (targetOS == OS.iOS) {
      // iOS 交叉编译配置
      cmakeArgs.addAll([
        '-DCMAKE_SYSTEM_NAME=iOS',
        '-DCMAKE_OSX_ARCHITECTURES=${targetArch.name}',
        '-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0',
        '-DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO',
      ]);
    } else if (targetOS == OS.macOS) {
      // macOS 配置
      final archFlag = switch (targetArch.name) {
        'arm64' => 'arm64',
        'x64' => 'x86_64',
        _ => targetArch.name,
      };
      cmakeArgs.addAll([
        '-DCMAKE_OSX_ARCHITECTURES=$archFlag',
        '-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0',
        // M1/M2 设备开启 Metal 支持
        '-DGGML_METAL=ON',
        '-DGGML_METAL_EMBED_LIBRARY=ON',
      ]);
    } else if (targetOS == OS.windows) {
      // Windows 配置
      if (targetArch == Architecture.arm64) {
        cmakeArgs.add('-A ARM64');
      }
      cmakeArgs.add('-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON');
    } else if (targetOS == OS.linux) {
      // Linux 配置
      cmakeArgs.add('-DCMAKE_POSITION_INDEPENDENT_CODE=ON');
      // 如果可用，启用 CUDA 支持
      cmakeArgs.add('-DGGML_CUDA=OFF'); // 默认关闭，需要时可开启
    }

    // 3. 执行 CMake 配置与构建
    print('📦 Configuring CMake...');
    await _runCommand('cmake', cmakeArgs);

    print('🔨 Building llama library...');
    await _runCommand('cmake', ['--build', buildDir.path, '--config', 'Release', '--parallel']);

    // 4. 定位生成的库文件
    final libName = _getLibraryName(targetOS);
    final libFile = _findLibrary(buildDir, libName, targetOS);

    if (libFile == null) {
      throw Exception('❌ Failed to find built library: $libName in ${buildDir.path}');
    }

    // 5. 注册资产 - 供 ffigen 生成的绑定使用
    output.assets.code.add(
      CodeAsset(package: 'llama_native', name: 'llama_native', linkMode: DynamicLoadingBundled(), file: libFile.uri),
    );

    // 6. 复制所有依赖的 dylib 文件（macOS 需要）- 排除主库避免重复
    if (targetOS == OS.macOS) {
      final binDir = p.join(buildDir.path, 'bin');
      final allDylibs = Directory(
        binDir,
      ).listSync().whereType<File>().where((f) => f.path.endsWith('.dylib') && f.path != libFile.path);

      for (final dylib in allDylibs) {
        output.assets.code.add(
          CodeAsset(
            package: 'llama_native',
            name: p.basename(dylib.path),
            linkMode: DynamicLoadingBundled(),
            file: dylib.uri,
          ),
        );
      }
      print('📦 Registered ${allDylibs.length} dependency dylib files for macOS');
    }

    // 7. 添加依赖文件（用于增量构建）
    output.dependencies.add(Uri.file(p.join(packageRoot, 'src', 'llama.cpp', 'CMakeLists.txt')));

    print('✅ Build Completed: ${libFile.path}');
  });
}

/// 辅助函数：执行系统命令
Future<void> _runCommand(String executable, List<String> arguments) async {
  final result = await Process.run(executable, arguments);
  if (result.exitCode != 0) {
    print('STDOUT: ${result.stdout}');
    print('STDERR: ${result.stderr}');
    throw ProcessException(executable, arguments, result.stderr, result.exitCode);
  }
}

/// 辅助函数：获取不同平台的库文件名
String _getLibraryName(OS os) {
  if (os == OS.android) return 'libllama.so';
  if (os == OS.iOS) return 'libllama.dylib';
  if (os == OS.macOS) return 'libllama.dylib';
  if (os == OS.linux) return 'libllama.so';
  if (os == OS.windows) return 'llama.dll';
  if (os == OS.fuchsia) return 'libllama.so';
  throw UnsupportedError('Unsupported OS: ${os.name}');
}

/// 辅助函数：映射 Android ABI
String _mapAndroidAbi(Architecture arch) {
  if (arch.name == 'arm') return 'armeabi-v7a';
  if (arch.name == 'arm64') return 'arm64-v8a';
  if (arch.name == 'ia32') return 'x86';
  if (arch.name == 'x64') return 'x86_64';
  if (arch.name == 'riscv64') return 'riscv64';
  throw UnsupportedError('Unsupported Android architecture: ${arch.name}');
}

/// 辅助函数：在目录下递归查找文件
File? _findLibrary(Directory dir, String fileName, OS os) {
  // 直接检查 bin 目录（llama.cpp 默认输出位置）
  final binDir = Directory(p.join(dir.path, 'bin'));
  if (binDir.existsSync()) {
    final directMatch = File(p.join(binDir.path, fileName));
    if (directMatch.existsSync()) {
      return directMatch;
    }
  }

  // 递归查找作为后备
  for (var entity in dir.listSync(recursive: true)) {
    if (entity is File && p.basename(entity.path) == fileName) {
      return entity;
    }
  }

  return null;
}
