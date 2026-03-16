import 'dart:io';
import 'package:archive/archive_io.dart';
import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:path/path.dart' as p;

/// 平台信息配置
class PlatformConfig {
  final String os;
  final String arch;
  final bool isSimulator;

  PlatformConfig({required this.os, required this.arch, this.isSimulator = false});
}

void main(List<String> args) async {
  await build(args, (input, output) async {
    final targetOS = input.config.code.targetOS.name;
    final targetArch = input.config.code.targetArchitecture.name;
    final packageRoot = input.packageRoot.toFilePath();

    print('\n🚀 ========== 开始构建 llama_native ==========');

    // 1. 读取版本号
    final llamaCppVersion = getLlamaCppVersion(packageRoot);

    // 2. 切换 llama.cpp 到指定版本
    print('\n📋 步骤 1: 切换 llama.cpp 版本');
    await checkoutLlamaCppVersion(packageRoot, llamaCppVersion);

    // 3. 生成 ffigen 绑定
    print('\n📋 步骤 2: 生成 FFI 绑定');
    await runFfigen(packageRoot);

    // Web 平台不支持
    if (targetOS == 'web') {
      print('⚠️  Web 平台不支持，跳过');
      return;
    }

    // 判断是否为模拟器
    bool isSimulator = false;
    if (targetOS == 'ios') {
      isSimulator = targetArch == 'x86_64';
    }

    // 创建平台配置
    final platformConfig = PlatformConfig(os: targetOS, arch: targetArch, isSimulator: isSimulator);

    // 处理该平台
    final libFile = await processPlatform(platformConfig, packageRoot, llamaCppVersion);

    // 注册资产
    output.assets.code.add(
      CodeAsset(package: 'llama_native', name: 'llama_native', linkMode: DynamicLoadingBundled(), file: libFile.uri),
    );

    print('\n✅ ========== 构建完成：${libFile.path} ==========');
  });
}

/// 从 pubspec.yaml 读取 llama.cpp 版本号
String getLlamaCppVersion(String packageRoot) {
  // 方式 1: 从环境变量读取（CI/CD 时使用）
  final envVersion = Platform.environment['LLAMA_CPP_VERSION'];
  if (envVersion != null && envVersion.isNotEmpty) {
    print('📌 使用环境变量 LLAMA_CPP_VERSION: $envVersion');
    return envVersion;
  }

  // 方式 2: 从 pubspec.yaml 读取
  try {
    final pubspecFile = File(p.join(packageRoot, 'pubspec.yaml'));
    if (pubspecFile.existsSync()) {
      final content = pubspecFile.readAsStringSync();
      final lines = content.split('\n');

      for (final line in lines) {
        if (line.startsWith('llama_cpp_version:')) {
          final version = line.split(':').last.trim();
          if (version.isNotEmpty) {
            print('📌 从 pubspec.yaml 读取 llama_cpp_version: $version');
            return version;
          }
        }
      }
    }
  } catch (e) {
    print('⚠️  读取 pubspec.yaml 失败：$e');
  }

  // 默认版本号
  print('⚠️  未找到 llama_cpp_version，使用默认值：b8369');
  return 'b8369';
}

/// 切换 llama.cpp 到指定版本
Future<void> checkoutLlamaCppVersion(String packageRoot, String version) async {
  final llamaCppDir = Directory(p.join(packageRoot, 'src', 'llama.cpp'));

  if (!llamaCppDir.existsSync()) {
    throw Exception('❌ llama.cpp 目录不存在：${llamaCppDir.path}');
  }

  print('🔄 准备切换 llama.cpp 到版本：$version');

  try {
    // 检查是否是 git 仓库
    final gitDir = Directory(p.join(llamaCppDir.path, '.git'));
    if (!gitDir.existsSync()) {
      print('⚠️  llama.cpp 不是 git 仓库，跳过版本切换');
      return;
    }

    // 执行 git checkout
    print('📍 正在执行 git checkout $version');
    final checkoutResult = await Process.run(
      'git',
      ['checkout', version],
      workingDirectory: llamaCppDir.path,
      runInShell: true,
    );

    if (checkoutResult.exitCode != 0) {
      print('⚠️  git checkout 失败：${checkoutResult.stderr}');
      print('ℹ️  尝试先拉取最新代码...');

      // 尝试先拉取
      final pullResult = await Process.run(
        'git',
        ['fetch', 'origin'],
        workingDirectory: llamaCppDir.path,
        runInShell: true,
      );

      if (pullResult.exitCode == 0) {
        // 再次尝试 checkout
        final retryResult = await Process.run(
          'git',
          ['checkout', version],
          workingDirectory: llamaCppDir.path,
          runInShell: true,
        );

        if (retryResult.exitCode != 0) {
          throw Exception('❌ 无法切换到版本 $version: ${retryResult.stderr}');
        }
      } else {
        throw Exception('❌ git fetch 失败：${pullResult.stderr}');
      }
    }

    print('✅ 成功切换到 llama.cpp 版本：$version');

    // 初始化子模块
    print('📦 初始化子模块...');
    final submoduleResult = await Process.run(
      'git',
      ['submodule', 'update', '--init', '--recursive'],
      workingDirectory: llamaCppDir.path,
      runInShell: true,
    );

    if (submoduleResult.exitCode != 0) {
      print('⚠️  子模块初始化警告：${submoduleResult.stderr}');
    } else {
      print('✅ 子模块初始化完成');
    }
  } catch (e) {
    print('❌ 切换 llama.cpp 版本失败：$e');
    rethrow;
  }
}

/// 运行 dart ffigen 生成绑定
Future<void> runFfigen(String packageRoot) async {
  print('🔧 正在运行 dart ffigen 生成绑定...');

  try {
    final result = await Process.run(
      'dart',
      ['run', 'ffigen', '--config', 'ffigen.yaml'],
      workingDirectory: packageRoot,
      runInShell: true,
    );

    if (result.exitCode != 0) {
      print('⚠️  ffigen 执行警告：');
      if (result.stdout.toString().isNotEmpty) {
        print('STDOUT: ${result.stdout}');
      }
      if (result.stderr.toString().isNotEmpty) {
        print('STDERR: ${result.stderr}');
      }
    } else {
      print('✅ ffigen 绑定生成成功');
      if (result.stdout.toString().isNotEmpty) {
        print('📝 ${result.stdout}');
      }
    }
  } catch (e) {
    print('❌ 运行 ffigen 失败：$e');
    print('ℹ️  请确保已安装 ffigen: dart pub global activate ffigen');
    rethrow;
  }
}

/// 获取二进制文件存储目录 .binary/{tag}/
Directory getBinaryDirectory(String packageRoot, String tagName) {
  return Directory(p.join(packageRoot, '.binaries', tagName));
}

/// 获取下载 URL
String getDownloadUrl(PlatformConfig config, String tagName) {
  final artifactName = _getArtifactName(config, tagName);

  return 'https://github.com/Ant1mage/llama_native/releases/download/$tagName/$artifactName';
}

/// 生成 artifact 文件名
String _getArtifactName(PlatformConfig config, String tagName) {
  if (config.os == 'ios') {
    return config.isSimulator ? 'llama-$tagName-ios-simulator.tar.gz' : 'llama-$tagName-ios-device.tar.gz';
  } else if (config.os == 'macos') {
    return 'llama-$tagName-macos.tar.gz';
  } else if (config.os == 'android') {
    return 'llama-$tagName-android-${config.arch}.tar.gz';
  } else if (config.os == 'linux') {
    return 'llama-$tagName-linux-x64.tar.gz';
  } else if (config.os == 'windows') {
    return 'llama-$tagName-windows-x64.tar.gz';
  }
  throw UnsupportedError('Unsupported platform: ${config.os}');
}

/// 下载文件
Future<File> downloadFile(String url, String destinationPath) async {
  print('📥 开始下载：$url');
  final client = HttpClient();
  try {
    final request = await client.getUrl(Uri.parse(url));
    final response = await request.close();

    if (response.statusCode != 200) {
      throw Exception('下载失败：HTTP ${response.statusCode}');
    }

    final file = File(destinationPath);
    await file.parent.create(recursive: true);
    await response.pipe(file.openWrite());

    print('✅ 下载完成：$destinationPath');
    return file;
  } finally {
    client.close();
  }
}

/// 解压 tar.gz 文件
Future<void> extractArchive(File archiveFile, String extractPath) async {
  print('📦 开始解压：${archiveFile.path}');
  final bytes = await archiveFile.readAsBytes();
  final archive = TarDecoder().decodeBytes(bytes);

  for (final file in archive) {
    final filePath = p.join(extractPath, file.name);

    if (file.isFile) {
      final outFile = File(filePath);
      await outFile.parent.create(recursive: true);
      await outFile.writeAsBytes(file.content);
      // 保留可执行权限
      if (file.mode & 0x111 != 0) {
        await Process.run('chmod', ['+x', filePath]);
      }
    } else {
      await Directory(filePath).create(recursive: true);
    }
  }

  print('✅ 解压完成：$extractPath');
}

/// 查找并处理库文件
/// 所有平台统一使用动态库：.dylib (macOS/iOS), .so (Android/Linux), .dll (Windows)
Future<File?> findAndCopyLibrary(String extractPath, String targetOS, String destDir) async {
  final extractDir = Directory(extractPath);

  if (!await extractDir.exists()) {
    return null;
  }

  final files = extractDir.listSync(recursive: true);
  final libName = _getLibraryName(targetOS);

  // 查找动态库文件
  for (final entity in files) {
    if (entity is File) {
      final ext = p.extension(entity.path);
      final basename = p.basename(entity.path);

      final isSharedLib = ext == '.dylib' || ext == '.so' || ext == '.dll';
      final isLlamaLib = basename.contains('llama') || basename.startsWith('lib');

      if (isSharedLib && isLlamaLib) {
        final newPath = p.join(destDir, libName);
        print('📋 复制库文件：${p.basename(entity.path)} -> $libName');
        await File(newPath).parent.create(recursive: true);
        await entity.copy(newPath);
        return File(newPath);
      }
    }
  }

  return null;
}

/// 获取不同平台的库文件名
String _getLibraryName(String os) {
  if (os == 'android' || os == 'linux') return 'libllama.so';
  if (os == 'ios' || os == 'macos') return 'libllama.dylib';
  if (os == 'windows') return 'llama.dll';
  throw UnsupportedError('Unsupported OS: $os');
}

/// 处理单个平台
Future<File> processPlatform(PlatformConfig config, String packageRoot, String tagName) async {
  print('\n========== 处理平台：${config.os}_${config.arch}${config.isSimulator ? '_simulator' : ''} ==========');

  // 获取二进制存储目录 .binary/{tag}/
  final binaryDir = getBinaryDirectory(packageRoot, tagName);
  final platformDirName = '${config.os}_${config.arch}${config.isSimulator ? '_simulator' : ''}';
  final platformDir = Directory(p.join(binaryDir.path, platformDirName));

  // 检查是否已下载
  final cachedLib = await _checkCachedLibrary(platformDir.path, config.os);
  if (cachedLib != null) {
    print('✅ 使用缓存的库文件：${cachedLib.path}');
    return cachedLib;
  }

  // 获取下载链接
  final downloadUrl = getDownloadUrl(config, tagName);

  // 创建下载和解压目录
  final archivePath = p.join(binaryDir.path, 'downloads', _getArtifactName(config, tagName));
  final extractPath = p.join(binaryDir.path, 'extracted', platformDirName);

  try {
    // 创建目录
    await platformDir.create(recursive: true);

    // 下载
    final archiveFile = await downloadFile(downloadUrl, archivePath);

    // 解压
    await extractArchive(archiveFile, extractPath);

    // 查找并复制库文件到目标目录
    final libFile = await findAndCopyLibrary(extractPath, config.os, platformDir.path);

    if (libFile == null) {
      throw Exception('❌ 未找到库文件');
    }

    print('✅ 平台 ${config.os}_${config.arch} 处理完成');
    print('📂 库文件位置：${libFile.path}');
    return libFile;
  } catch (e) {
    print('❌ 处理平台 ${config.os}_${config.arch} 失败：$e');
    rethrow;
  }
}

/// 检查缓存的库文件
Future<File?> _checkCachedLibrary(String platformDir, String targetOS) async {
  final dir = Directory(platformDir);

  if (!await dir.exists()) {
    return null;
  }

  final libName = _getLibraryName(targetOS);
  final libPath = p.join(platformDir, libName);
  if (await File(libPath).exists()) {
    return File(libPath);
  }

  return null;
}
