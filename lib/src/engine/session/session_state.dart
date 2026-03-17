import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:crypto/crypto.dart';
import 'package:llama_native/src/engine/cache/kv_cache_snapshot.dart';
import 'package:llama_native/src/utils/disposable.dart';
import 'package:llama_native/src/engine/context/llama_context.dart';
import 'package:llama_native/src/log/logger.dart';
import 'package:llama_native/src/engine/exceptions/llama_exceptions.dart';

/// 持久化快照
///
/// 负责：
/// - 会话内存序列化
/// - 跨平台同步与恢复
/// - 管理 state_id, buffer_size
class SessionState implements Disposable {
  final Logger _logger;

  /// 会话 ID
  final String sessionId;

  /// 创建时间
  final DateTime createdAt;

  /// KV Cache 快照
  KVCacheSnapshot? _cacheSnapshot;

  /// 序列化数据
  Uint8List? _serializedData;

  /// 是否已释放
  bool _disposed = false;

  /// 创建会话状态
  SessionState({String? sessionId, DateTime? createdAt})
    : sessionId = sessionId ?? _generateSessionId(),
      createdAt = createdAt ?? DateTime.now(),
      _logger = Logger('SessionState');

  /// 生成会话 ID
  static String _generateSessionId() {
    final timestamp = DateTime.now().millisecondsSinceEpoch.toString();
    final hash = sha256.convert(timestamp.codeUnits);
    return hash.toString().substring(0, 16);
  }

  /// 从上下文创建快照
  Future<void> captureFrom(LlamaContext context) async {
    if (_disposed) throw StateError('SessionState is disposed');

    _logger.info('Capturing session state from context');

    _cacheSnapshot = KVCacheSnapshot.fromContext(context.ctxPtr, keepPrefix: context.keepPrefix);

    _logger.info('Session captured: ${context.nPast} tokens');
  }

  /// 恢复到上下文
  Future<void> restoreTo(LlamaContext context) async {
    if (_disposed) throw StateError('SessionState is disposed');

    if (_cacheSnapshot == null) {
      throw StateError('No snapshot to restore');
    }

    _logger.info('Restoring session state to context');

    final success = _cacheSnapshot!.restoreTo(context.ctxPtr);
    if (success) {
      _cacheSnapshot!.restoreToManager(context.kvCache);
      _logger.info('Session restored: ${context.nPast} tokens');
    } else {
      _logger.warning('Failed to restore session state');
    }
  }

  /// 序列化为字节
  Future<Uint8List> serialize() async {
    if (_disposed) throw StateError('SessionState is disposed');

    _logger.debug('Serializing session state');

    final data = <String, dynamic>{
      'session_id': sessionId,
      'created_at': createdAt.toIso8601String(),
      'n_past': _cacheSnapshot?.nPast ?? 0,
      'keep_prefix': _cacheSnapshot?.keepPrefix ?? 0,
      'state_data': _cacheSnapshot?.stateData != null ? base64Encode(_cacheSnapshot!.stateData!) : '',
    };

    final jsonStr = jsonEncode(data);
    _serializedData = Uint8List.fromList(jsonStr.codeUnits);

    _logger.debug('Serialized to ${_serializedData!.length} bytes');
    return _serializedData!;
  }

  /// 从字节反序列化
  static Future<SessionState> deserialize(Uint8List data) async {
    try {
      final jsonStr = String.fromCharCodes(data);
      final parsed = jsonDecode(jsonStr) as Map<String, dynamic>;

      final session = SessionState(
        sessionId: parsed['session_id'] as String,
        createdAt: DateTime.parse(parsed['created_at'] as String),
      );

      final nPast = parsed['n_past'] as int;
      final keepPrefix = parsed['keep_prefix'] as int;
      final stateDataStr = parsed['state_data'] as String?;
      final stateData = stateDataStr != null && stateDataStr.isNotEmpty ? base64Decode(stateDataStr) : null;

      session._cacheSnapshot = KVCacheSnapshot(
        nPast: nPast,
        keepPrefix: keepPrefix,
        windowSize: null,
        stateData: stateData,
      );

      session._serializedData = data;
      return session;
    } catch (e) {
      throw LlamaException.session('Failed to deserialize: $e');
    }
  }

  /// 保存到文件
  Future<File> saveToFile(String path) async {
    if (_disposed) throw StateError('SessionState is disposed');

    _logger.info('Saving session to file: $path');

    final data = await serialize();
    final file = File(path);

    // 确保目录存在
    final dir = file.parent;
    if (!dir.existsSync()) {
      dir.createSync(recursive: true);
    }

    await file.writeAsBytes(data);

    _logger.info('Session saved: ${file.path}');
    return file;
  }

  /// 从文件加载
  static Future<SessionState> loadFromFile(String path) async {
    final file = File(path);

    if (!file.existsSync()) {
      throw FileSystemException('Session file not found', path);
    }

    final data = await file.readAsBytes();
    return deserialize(data);
  }

  /// 获取缓冲区大小
  int get bufferSize => _serializedData?.length ?? 0;

  /// 是否有序列化数据
  bool get hasData => _serializedData != null;

  /// 清除数据
  void clear() {
    _cacheSnapshot = null;
    _serializedData = null;
    _logger.debug('Session data cleared');
  }

  @override
  void dispose() {
    if (_disposed) return;

    _logger.debug('Disposing SessionState');
    clear();
    _disposed = true;
  }

  /// 是否已释放
  bool get isDisposed => _disposed;
}

/// 会话管理器 (管理多个会话)
class SessionManager {
  final Map<String, SessionState> _sessions = {};
  final Logger _logger = Logger('SessionManager');

  /// 创建新会话
  SessionState createSession() {
    final session = SessionState();
    _sessions[session.sessionId] = session;
    _logger.debug('Created session: ${session.sessionId}');
    return session;
  }

  /// 获取会话
  SessionState? getSession(String sessionId) {
    return _sessions[sessionId];
  }

  /// 删除会话
  void removeSession(String sessionId) {
    final session = _sessions.remove(sessionId);
    session?.dispose();
    _logger.debug('Removed session: $sessionId');
  }

  /// 列出所有会话
  List<String> listSessions() {
    return _sessions.keys.toList();
  }

  /// 清空所有会话
  void clearAll() {
    for (final session in _sessions.values) {
      session.dispose();
    }
    _sessions.clear();
    _logger.debug('All sessions cleared');
  }

  /// 保存所有会话到目录
  Future<void> saveAllToDirectory(String dirPath) async {
    final dir = Directory(dirPath);
    if (!dir.existsSync()) {
      dir.createSync(recursive: true);
    }

    for (final session in _sessions.values) {
      final path = '${dirPath}/${session.sessionId}.session';
      await session.saveToFile(path);
    }

    _logger.info('Saved ${_sessions.length} sessions');
  }

  /// 从目录加载所有会话
  Future<void> loadAllFromDirectory(String dirPath) async {
    final dir = Directory(dirPath);
    if (!dir.existsSync()) return;

    final files = dir.listSync().whereType<File>().where((f) => f.path.endsWith('.session'));

    for (final file in files) {
      try {
        final session = await SessionState.loadFromFile(file.path);
        _sessions[session.sessionId] = session;
      } catch (e) {
        _logger.error('Failed to load session: ${file.path}', e);
      }
    }

    _logger.info('Loaded ${_sessions.length} sessions');
  }
}
