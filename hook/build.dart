import 'dart:io';
import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:path/path.dart' as p;

class PlatformConfig {
  final String os;
  final String arch;
  final bool isSimulator;

  PlatformConfig({required this.os, required this.arch, this.isSimulator = false});

  String get normalizedArch {
    final lowerArch = arch.toLowerCase();
    if (lowerArch == 'arm64' || lowerArch == 'aarch64') {
      return 'arm64';
    } else if (lowerArch == 'x86_64' || lowerArch == 'x64' || lowerArch == 'amd64') {
      return 'x86_64';
    } else if (lowerArch == 'arm' || lowerArch == 'armeabi-v7a' || lowerArch == 'armv7') {
      return 'arm';
    }
    return arch;
  }

  String get platformDisplayName {
    if (os == 'ios' && isSimulator) {
      return 'ios-simulator-$normalizedArch';
    }
    return '$os-$normalizedArch';
  }

  String get dirName {
    if (os == 'ios' && isSimulator) {
      return 'ios-simulator-$normalizedArch';
    }
    return '$os-$normalizedArch';
  }

  String get artifactName {
    if (os == 'ios' && isSimulator) {
      return 'ios-simulator-$normalizedArch.tar.gz';
    }
    return '$os-$normalizedArch.tar.gz';
  }
}

void main(List<String> args) async {
  await build(args, (input, output) async {
    final targetOS = input.config.code.targetOS.name;
    final targetArch = input.config.code.targetArchitecture.name;
    final packageRoot = input.packageRoot.toFilePath();

    print('\n🚀 ========== 开始构建 llama_native ==========');

    final llamaCppVersion = getLlamaCppVersion(packageRoot);

    print('\n📋 步骤 1: 切换 llama.cpp 版本');
    await checkoutLlamaCppVersion(packageRoot, llamaCppVersion);

    print('\n📋 步骤 2: 生成 FFI 绑定');
    await runFfigen(packageRoot);

    if (targetOS == 'web') {
      print('⚠️  Web 平台不支持，跳过');
      return;
    }

    bool isSimulator = false;
    if (targetOS == 'ios') {
      isSimulator = input.config.code.iOS.targetSdk == IOSSdk.iPhoneSimulator;
    }

    print('🎯 目标平台: $targetOS, 目标架构: $targetArch');

    final platformConfig = PlatformConfig(os: targetOS, arch: targetArch, isSimulator: isSimulator);

    final libFiles = await processPlatform(platformConfig, packageRoot, llamaCppVersion);

    final mainLibName = _getMainLibraryName(targetOS);
    File? mainLib;
    for (final lib in libFiles) {
      if (p.basename(lib.path) == mainLibName) {
        mainLib = lib;
        break;
      }
    }

    if (mainLib == null) {
      throw Exception('❌ 未找到主库文件：$mainLibName');
    }

    output.assets.code.add(
      CodeAsset(package: 'llama_native', name: 'llama_native', linkMode: DynamicLoadingBundled(), file: mainLib.uri),
    );

    for (final lib in libFiles) {
      if (p.basename(lib.path) != mainLibName) {
        output.assets.code.add(
          CodeAsset(
            package: 'llama_native',
            name: 'native_${p.basenameWithoutExtension(lib.path)}',
            linkMode: DynamicLoadingBundled(),
            file: lib.uri,
          ),
        );
      }
    }

    print('\n✅ ========== 构建完成：${libFiles.length} 个库文件 ==========');
    print('📂 主库：${mainLib.path}');
    for (final lib in libFiles) {
      if (p.basename(lib.path) != mainLibName) {
        print('   依赖：${p.basename(lib.path)}');
      }
    }
  });
}

String getLlamaCppVersion(String packageRoot) {
  final envVersion = Platform.environment['LLAMA_CPP_VERSION'];
  if (envVersion != null && envVersion.isNotEmpty) {
    print('📌 使用环境变量 LLAMA_CPP_VERSION: $envVersion');
    return envVersion;
  }

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

  print('⚠️  未找到 llama_cpp_version，使用默认值：b8369');
  return 'b8369';
}

Future<void> checkoutLlamaCppVersion(String packageRoot, String version) async {
  final llamaCppDir = Directory(p.join(packageRoot, 'src', 'llama.cpp'));

  if (!llamaCppDir.existsSync()) {
    throw Exception('❌ llama.cpp 目录不存在：${llamaCppDir.path}');
  }

  print('🔄 准备切换 llama.cpp 到版本：$version');

  try {
    final gitDir = Directory(p.join(llamaCppDir.path, '.git'));
    if (!gitDir.existsSync()) {
      print('⚠️  llama.cpp 不是 git 仓库，跳过版本切换');
      return;
    }

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

      final pullResult = await Process.run(
        'git',
        ['fetch', 'origin'],
        workingDirectory: llamaCppDir.path,
        runInShell: true,
      );

      if (pullResult.exitCode == 0) {
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

String getDownloadUrl(PlatformConfig config, String tagName) {
  return 'https://github.com/Ant1mage/llama_native/releases/download/$tagName/${config.artifactName}';
}

Future<File> downloadFile(String url, String destinationPath) async {
  print('📥 开始下载：$url');
  final client = HttpClient();
  try {
    // 获取环境变量
    final token = Platform.environment['GITHUB_TOKEN'];
    if (token == null || token.isEmpty) {
      throw Exception("❌ 错误：找不到 GITHUB_TOKEN，请先在终端执行 export 设置它。");
    }

    print("✅ 已成功加载 Token，可以开始下载私有 Release。");
    final request = await client.getUrl(Uri.parse(url));
    request.headers.add('Authorization', 'Bearer $token');
    request.headers.add('Accept', 'application/octet-stream');
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

Future<void> extractArchive(File archiveFile, String extractPath) async {
  print('📦 开始解压：${archiveFile.path}');
  await Directory(extractPath).create(recursive: true);

  final result = await Process.run('tar', ['-xzf', archiveFile.path, '-C', extractPath]);

  if (result.exitCode != 0) {
    throw Exception('解压失败：${result.stderr}');
  }

  print('✅ 解压完成：$extractPath');
}

List<String> _getLibExtensions(String os) {
  if (os == 'macos' || os == 'ios') return ['.dylib'];
  if (os == 'android' || os == 'linux') return ['.so'];
  if (os == 'windows') return ['.dll'];
  throw UnsupportedError('Unsupported OS: $os');
}

Future<List<File>> findAllLibraries(String extractPath, String targetOS) async {
  final extractDir = Directory(extractPath);

  if (!await extractDir.exists()) {
    return [];
  }

  final files = extractDir.listSync(recursive: true);
  final libExtensions = _getLibExtensions(targetOS);
  final libraries = <File>[];

  for (final entity in files) {
    if (entity is File) {
      final ext = p.extension(entity.path);
      if (libExtensions.contains(ext)) {
        libraries.add(entity);
      }
    }
  }

  return libraries;
}

String _getMainLibraryName(String os) {
  if (os == 'android' || os == 'linux') return 'libllama.so';
  if (os == 'ios' || os == 'macos') return 'libllama.dylib';
  if (os == 'windows') return 'llama.dll';
  throw UnsupportedError('Unsupported OS: $os');
}

Future<List<File>> processPlatform(PlatformConfig config, String packageRoot, String tagName) async {
  print('\n========== 处理平台：${config.platformDisplayName} ==========');

  final platformDir = Directory(p.join(packageRoot, '.binaries', tagName, config.dirName));

  final cachedLibs = await _checkCachedLibraries(platformDir.path, config.os);
  if (cachedLibs.isNotEmpty) {
    print('✅ 使用缓存的库文件：${cachedLibs.length} 个');
    return cachedLibs;
  }

  final downloadUrl = getDownloadUrl(config, tagName);
  final archivePath = p.join(platformDir.path, config.artifactName);

  try {
    await platformDir.create(recursive: true);

    final archiveFile = await downloadFile(downloadUrl, archivePath);

    await extractArchive(archiveFile, platformDir.path);

    final libFiles = await findAllLibraries(platformDir.path, config.os);

    if (libFiles.isEmpty) {
      throw Exception('❌ 未找到库文件');
    }

    await archiveFile.delete();

    print('✅ 平台 ${config.platformDisplayName} 处理完成');
    print('📂 库文件数量：${libFiles.length}');
    for (final lib in libFiles) {
      print('   - ${p.basename(lib.path)}');
    }
    return libFiles;
  } catch (e) {
    print('❌ 处理平台 ${config.platformDisplayName} 失败：$e');
    rethrow;
  }
}

Future<List<File>> _checkCachedLibraries(String platformDir, String targetOS) async {
  final dir = Directory(platformDir);

  if (!await dir.exists()) {
    return [];
  }

  final mainLibName = _getMainLibraryName(targetOS);
  final mainLibPath = p.join(platformDir, mainLibName);
  if (!await File(mainLibPath).exists()) {
    return [];
  }

  return await findAllLibraries(platformDir, targetOS);
}
