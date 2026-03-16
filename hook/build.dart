import 'dart:io';
import 'package:archive/archive.dart';
import 'package:hooks/hooks.dart';
import 'package:http/http.dart' as http;
import 'package:code_assets/code_assets.dart';

/// ============================================================
/// [LLAMA_NATIVE] 核心配置 - 修改此处即可驱动全流程
/// ============================================================
const String LLAMA_TAG = "b8369";
const String PACKAGE_NAME = 'llama_native';
const String ASSET_ID = 'llama_native';

/// ============================================================

void main(List<String> args) async {
  await build(args, (config, output) async {
    final os = config.config.code.targetOS;
    final arch = config.config.code.targetArchitecture;
    final isSimulator = _isSimulator(os, arch);

    // 1. 同步 Git 子模块：确保 vendor/llama.cpp 源码处于正确的 TAG
    log("同步 llama.cpp 子模块");
    await _syncGitSubmodule(config.packageRoot);

    // 2. 自动化 Bindings：确保生成的 dart 代码与当前 C 头文件一致
    log("生成 llama.cpp 的 bindings");
    await _runFfigen(config.packageRoot);

    // 3. 确定存储目录与最终路径：.binaries/b4600/os_arch.so
    final binariesDir = Directory.fromUri(config.packageRoot.resolve('.binaries/$LLAMA_TAG/'));
    final libFileName = _getLibFileName(os, arch, isSimulator);
    final File finalLibFile = File('${binariesDir.path}/$libFileName');

    // 4. 供应逻辑：如果文件不存在则下载，存在则跳过直接挂载
    if (!finalLibFile.existsSync()) {
      log('[$PACKAGE_NAME] 库文件不存在，准备下载 $LLAMA_TAG ($os-$arch)...');
      final url = _getDownloadUrl(LLAMA_TAG, os, arch, isSimulator);
      await _downloadAndProvision(url, finalLibFile, binariesDir, os, arch, isSimulator);
    } else {
      log('[$PACKAGE_NAME] 检测到本地库已就绪，直接挂载: $libFileName');
    }

    // 5. 核心：通过 Native Assets 机制挂载
    _registerAsset(output, os, arch, isSimulator, binariesDir);
  });
}

/// --- 1. Git 子模块管理 ---
Future<void> _syncGitSubmodule(Uri root) async {
  final path = root.resolve('src/llama.cpp').toFilePath();
  final gitDir = Directory('$path/.git');

  // 如果子模块没初始化过，或者当前 HEAD 不是目标 TAG，则执行同步
  log('[$PACKAGE_NAME] 检查子模块状态...');

  final cmds = [
    ['submodule', 'update', '--init', '--recursive'],
    ['fetch', '--tags'],
    ['checkout', LLAMA_TAG],
  ];

  for (var cmd in cmds) {
    await Process.run('git', cmd, workingDirectory: path);
  }
}

/// --- 2. Ffigen 自动执行 ---
Future<void> _runFfigen(Uri root) async {
  // 注意：这里可以加入简单的缓存逻辑，比如判断头文件修改时间
  log('[$PACKAGE_NAME] 正在生成/刷新 FFI 绑定代码...');
  final res = await Process.run('dart', [
    'run',
    'ffigen',
    '--config',
    'ffigen.yaml',
  ], workingDirectory: root.toFilePath());
  if (res.exitCode != 0) {
    log('[$PACKAGE_NAME] ffigen 警告: ${res.stderr}');
  }
}

/// --- 3. 下载地址构建 (请根据你的存储规则输入) ---
String _getDownloadUrl(String tag, OS os, Architecture arch, bool isSimulator) {
  final String baseUrl = "https://github.com/ggml-org/llama.cpp/releases/download/$tag";

  String fileName;
  switch (os) {
    case OS.macOS:
      fileName = "llama-$tag-bin-macos-${arch == Architecture.arm64 ? 'arm64' : 'x64'}.tar.gz";
      break;
    case OS.iOS:
      // fileName = "";
      // break;
      throw UnsupportedError("[$PACKAGE_NAME] 尚未配置 $os 平台的下载地址");
    case OS.android:
      // fileName = "";
      // break;
      throw UnsupportedError("[$PACKAGE_NAME] 尚未配置 $os 平台的下载地址");
    case OS.windows:
      // fileName = "";
      // break;
      throw UnsupportedError("[$PACKAGE_NAME] 尚未配置 $os 平台的下载地址");
    default:
      throw UnsupportedError("[$PACKAGE_NAME] 尚未配置 $os 平台的下载地址");
  }
  return "$baseUrl/$fileName";
}

/// --- 4. 下载与解压核心 ---
Future<void> _downloadAndProvision(
  String url,
  File target,
  Directory tagDir,
  OS os,
  Architecture arch,
  bool sim,
) async {
  if (!tagDir.existsSync()) tagDir.createSync(recursive: true);

  final tmpFile = File('${tagDir.path}/download_tmp.archive');
  final String platformName = "${os.name}-${arch.name}${sim ? '_sim' : ''}";
  final Directory platformDir = Directory('${tagDir.path}/$platformName');

  try {
    log("正在下载至临时文件...");
    final resp = await http.get(Uri.parse(url));
    await tmpFile.writeAsBytes(resp.bodyBytes);

    log("开始解压...");
    final bytes = tmpFile.readAsBytesSync();
    Archive archive;
    if (url.endsWith('.zip')) {
      archive = ZipDecoder().decodeBytes(bytes);
    } else {
      archive = TarDecoder().decodeBytes(GZipDecoder().decodeBytes(bytes));
    }

    // 1. 先把所有内容解压到 tagDir 下
    for (final file in archive) {
      final String filename = file.name;
      if (file.isFile) {
        final data = file.content as List<int>;
        File('${tagDir.path}/$filename')
          ..createSync(recursive: true)
          ..writeAsBytesSync(data);
      } else {
        Directory('${tagDir.path}/$filename').createSync(recursive: true);
      }
    }

    // 2. 找到解压出来的那个原始文件夹（通常是列表里的第一个文件夹）
    // 比如：llama-b8369-bin-macos-arm64/
    final firstDir = tagDir.listSync().firstWhere((e) => e is Directory && !e.path.contains(platformName));

    // 3. 重命名为我们要的名字（如 macos-arm64）
    log("重命名文件夹: ${firstDir.path} -> ${platformDir.path}");
    if (platformDir.existsSync()) platformDir.deleteSync(recursive: true);
    await firstDir.rename(platformDir.path);

    log("处理完成，文件夹位于: ${platformDir.path}");
  } finally {
    if (tmpFile.existsSync()) tmpFile.deleteSync();
  }
}


/// --- 辅助工具 ---

bool _isNativeLib(String name) => name.endsWith('.so') || name.endsWith('.dylib') || name.endsWith('.dll');

String _getLibFileName(OS os, Architecture arch, bool sim) {
  final ext = os == OS.windows ? 'dll' : (os == OS.macOS || os == OS.iOS ? 'dylib' : 'so');
  final simTag = sim ? "_sim" : "";
  return "${os.name}_${arch.name}$simTag.$ext";
}

bool _isSimulator(OS os, Architecture arch) {
  if (os == OS.iOS) return arch == IOSSdk.iPhoneSimulator;
  if (os == OS.android) return arch == Architecture.x64;
  return false;
}

void _registerAsset(BuildOutputBuilder output, OS os, Architecture arch, bool sim, Directory tagDir) {
  final String platformName = "${os.name}-${arch.name}${sim ? '_sim' : ''}";
  // 假设 dylib 在重命名后的文件夹根目录下，或者在 bin 目录下
  // 你需要确保这个路径能定位到那个 .dylib
  final platformDir = Directory('${tagDir.path}/$platformName');

  // 自动寻找文件夹下的第一个 dylib 进行挂载
  final dylibFile = platformDir
      .listSync(recursive: true)
      .firstWhere((e) => e is File && _isNativeLib(e.path), orElse: () => throw "在 $platformName 文件夹中未找到 dylib");

  log("挂载动态库: ${dylibFile.path}");

  output.assets.code.add(
    CodeAsset(package: PACKAGE_NAME, name: ASSET_ID, linkMode: DynamicLoadingBundled(), file: dylibFile.uri),
  );
}

void log(String message) {
  stdout.writeln('===> $message');
}
