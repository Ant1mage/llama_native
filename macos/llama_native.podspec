#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint llama_native.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'llama_native'
  s.version          = '0.0.1'
  s.summary          = 'A new Flutter FFI plugin project.'
  s.description      = <<-DESC
A new Flutter FFI plugin project.
                       DESC
  s.homepage         = 'http://example.com'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'email@example.com' }

  # FFI plugin - library is built by hook/build.dart using CMake
  # No source files needed in podspec
  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'

  # The library will be loaded dynamically from the bundle
  s.preserve_paths   = ['**/*']

  # If your plugin requires a privacy manifest, for example if it collects user
  # data, update the PrivacyInfo.xcprivacy file to describe your plugin's
  # privacy impact, and then uncomment this line. For more information,
  # see https://developer.apple.com/documentation/bundleresources/privacy_manifest_files
  # s.resource_bundles = {'llama_native_privacy' => ['Resources/PrivacyInfo.xcprivacy']}

  s.dependency 'FlutterMacOS'

  s.platform = :osx, '10.11'
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES' }
  s.swift_version = '5.0'
end
