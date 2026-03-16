import 'dart:io';
import 'package:code_assets/code_assets.dart';
import 'package:test/test.dart';
import '../hook/build.dart' as build_hook;

void main() {
  test('build hook for macOS x64', () async {
    await testCodeBuildHook(
      mainMethod: build_hook.main,
      targetOS: OS.macOS,
      targetArchitecture: Architecture.x64,
      targetMacOSVersion: 13,
      check: (input, output) {
        print('Build input:');
        print('  targetOS: ${input.config.code.targetOS}');
        print('  targetArchitecture: ${input.config.code.targetArchitecture}');

        print('\nBuild output:');
        for (final asset in output.assets.code) {
          print('  asset: ${asset.id} -> ${asset.file}');
        }

        expect(output.assets.code.length, greaterThan(0));

        final asset = output.assets.code.first;
        expect(asset.id, equals('package:llama_native/llama_native'));
        expect(asset.file, isNotNull);
        expect(asset.file!.toFilePath(), contains('libllama.dylib'));
      },
    );
  });

  test('build hook for macOS arm64', () async {
    await testCodeBuildHook(
      mainMethod: build_hook.main,
      targetOS: OS.macOS,
      targetArchitecture: Architecture.arm64,
      targetMacOSVersion: 13,
      check: (input, output) {
        print('Build input:');
        print('  targetOS: ${input.config.code.targetOS}');
        print('  targetArchitecture: ${input.config.code.targetArchitecture}');

        print('\nBuild output:');
        for (final asset in output.assets.code) {
          print('  asset: ${asset.id} -> ${asset.file}');
        }

        expect(output.assets.code.length, greaterThan(0));
      },
    );
  });

  test('build hook for iOS simulator arm64', () async {
    await testCodeBuildHook(
      mainMethod: build_hook.main,
      targetOS: OS.iOS,
      targetArchitecture: Architecture.arm64,
      targetIOSSdk: IOSSdk.iPhoneSimulator,
      targetIOSVersion: 13,
      check: (input, output) {
        print('Build input:');
        print('  targetOS: ${input.config.code.targetOS}');
        print('  targetArchitecture: ${input.config.code.targetArchitecture}');
        print('  targetSdk: ${input.config.code.iOS.targetSdk}');

        print('\nBuild output:');
        for (final asset in output.assets.code) {
          print('  asset: ${asset.id} -> ${asset.file}');
        }

        expect(output.assets.code.length, greaterThan(0));
      },
    );
  });

  test('build hook for iOS simulator x64', () async {
    await testCodeBuildHook(
      mainMethod: build_hook.main,
      targetOS: OS.iOS,
      targetArchitecture: Architecture.x64,
      targetIOSSdk: IOSSdk.iPhoneSimulator,
      targetIOSVersion: 13,
      check: (input, output) {
        print('Build input:');
        print('  targetOS: ${input.config.code.targetOS}');
        print('  targetArchitecture: ${input.config.code.targetArchitecture}');
        print('  targetSdk: ${input.config.code.iOS.targetSdk}');

        print('\nBuild output:');
        for (final asset in output.assets.code) {
          print('  asset: ${asset.id} -> ${asset.file}');
        }

        expect(output.assets.code.length, greaterThan(0));
      },
    );
  });

  test('build hook for iOS device', () async {
    await testCodeBuildHook(
      mainMethod: build_hook.main,
      targetOS: OS.iOS,
      targetArchitecture: Architecture.arm64,
      targetIOSSdk: IOSSdk.iPhoneOS,
      targetIOSVersion: 13,
      check: (input, output) {
        print('Build input:');
        print('  targetOS: ${input.config.code.targetOS}');
        print('  targetArchitecture: ${input.config.code.targetArchitecture}');
        print('  targetSdk: ${input.config.code.iOS.targetSdk}');

        print('\nBuild output:');
        for (final asset in output.assets.code) {
          print('  asset: ${asset.id} -> ${asset.file}');
        }

        expect(output.assets.code.length, greaterThan(0));
      },
    );
  });

  test('build hook for Android arm64', () async {
    await testCodeBuildHook(
      mainMethod: build_hook.main,
      targetOS: OS.android,
      targetArchitecture: Architecture.arm64,
      targetAndroidNdkApi: 24,
      check: (input, output) {
        print('Build input:');
        print('  targetOS: ${input.config.code.targetOS}');
        print('  targetArchitecture: ${input.config.code.targetArchitecture}');

        print('\nBuild output:');
        for (final asset in output.assets.code) {
          print('  asset: ${asset.id} -> ${asset.file}');
        }

        expect(output.assets.code.length, greaterThan(0));
      },
    );
  });

  test('build hook for Android x64', () async {
    await testCodeBuildHook(
      mainMethod: build_hook.main,
      targetOS: OS.android,
      targetArchitecture: Architecture.x64,
      targetAndroidNdkApi: 24,
      check: (input, output) {
        print('Build input:');
        print('  targetOS: ${input.config.code.targetOS}');
        print('  targetArchitecture: ${input.config.code.targetArchitecture}');

        print('\nBuild output:');
        for (final asset in output.assets.code) {
          print('  asset: ${asset.id} -> ${asset.file}');
        }

        expect(output.assets.code.length, greaterThan(0));
      },
    );
  });
}
