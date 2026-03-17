import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'package:llama_native/src/llama_native_bindings.dart' as bindings;
import 'kv_cache_manager.dart';
import 'package:llama_native/src/log/logger.dart';

class KVCacheSnapshot {
  static final Logger _staticLogger = Logger('KVCacheSnapshot');

  final int nPast;
  final int keepPrefix;
  final int? windowSize;
  final Uint8List? stateData;

  KVCacheSnapshot({required this.nPast, required this.keepPrefix, this.windowSize, this.stateData});

  factory KVCacheSnapshot.fromContext(Pointer<bindings.llama_context> ctx, {int keepPrefix = 0}) {
    final stateSize = bindings.llama_state_get_size(ctx);
    _staticLogger.info('Capturing KV cache state: size=$stateSize bytes');

    Uint8List? stateData;
    if (stateSize > 0) {
      final buffer = calloc<Uint8>(stateSize);
      try {
        final written = bindings.llama_state_get_data(ctx, buffer.cast(), stateSize);
        if (written > 0) {
          final tempList = buffer.asTypedList(written);
          stateData = Uint8List.fromList(tempList);
        }
      } finally {
        calloc.free(buffer);
      }
    }

    return KVCacheSnapshot(nPast: 0, keepPrefix: keepPrefix, windowSize: null, stateData: stateData);
  }

  bool restoreTo(Pointer<bindings.llama_context> ctx) {
    if (stateData == null || stateData!.isEmpty) {
      _staticLogger.warning('No state data to restore');
      return false;
    }

    _staticLogger.info('Restoring KV cache state: ${stateData!.length} bytes');

    final buffer = calloc<Uint8>(stateData!.length);
    try {
      buffer.asTypedList(stateData!.length).setAll(0, stateData!);
      final read = bindings.llama_state_set_data(ctx, buffer.cast(), stateData!.length);
      _staticLogger.info('KV cache state restored: $read bytes');
      return read > 0;
    } finally {
      calloc.free(buffer);
    }
  }

  void restoreToManager(KVCacheManager manager) {
    if (manager.isDisposed) {
      throw StateError('KVCacheManager is disposed');
    }

    _staticLogger.info('Restoring KV cache snapshot to manager: n_past=$nPast, keep_prefix=$keepPrefix');

    manager.reset();
    manager.setKeepPrefix(keepPrefix);
    manager.addProcessed(nPast);

    _staticLogger.debug('KV cache manager state restored');
  }
}
