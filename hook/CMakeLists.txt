# Placeholder CMakeLists.txt for Android Gradle build
# The actual build is performed by hook/build.dart using CMake

cmake_minimum_required(VERSION 3.10)
project("llama_native" C)

# This file exists only to satisfy the Android Gradle Plugin's CMake requirement
# The real llama library is built by the Dart hooks system in hook/build.dart

# Create a dummy target to avoid build errors
add_library(llama_native SHARED IMPORTED)
