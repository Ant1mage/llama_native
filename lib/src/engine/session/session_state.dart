import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:crypto/crypto.dart';
import 'package:llama_native/src/engine/cache/kv_cache_snapshot.dart';
import 'package:llama_native/src/engine/context/llama_context.dart';
import 'package:llama_native/src/log/logger.dart';
import 'package:llama_native/src/engine/exceptions/llama_exceptions.dart';

class SessionState {
  final Logger _logger;

  final String sessionId;
  final DateTime createdAt;

  KVCacheSnapshot? _cacheSnapshot;
  Uint8List? _serializedData;
  bool _disposed = false;

  SessionState({String? sessionId, DateTime? createdAt})
    : sessionId = sessionId ?? _generateSessionId(),
      createdAt = createdAt ?? DateTime.now(),
      _logger = Logger('SessionState');

  static String _generateSessionId() {
    final timestamp = DateTime.now().millisecondsSinceEpoch.toString();
    final hash = sha256.convert(timestamp.codeUnits);
    return hash.toString().substring(0, 16);
  }

  Future<void> captureFrom(LlamaContext context) async {
    if (_disposed) throw StateError('SessionState已释放');

    _logger.info('从上下文捕获会话状态');

    _cacheSnapshot = KVCacheSnapshot.fromContext(context.ctxPtr, keepPrefix: context.keepPrefix);

    _logger.info('会话已捕获: ${context.nPast}个Token');
  }

  Future<void> restoreTo(LlamaContext context) async {
    if (_disposed) throw StateError('SessionState已释放');

    if (_cacheSnapshot == null) {
      throw StateError('没有快照可恢复');
    }

    _logger.info('恢复会话状态到上下文');

    final success = _cacheSnapshot!.restoreTo(context.ctxPtr);
    if (success) {
      _cacheSnapshot!.restoreToManager(context.kvCache);
      _logger.info('会话已恢复: ${context.nPast}个Token');
    } else {
      _logger.warning('恢复会话状态失败');
    }
  }

  Future<Uint8List> serialize() async {
    if (_disposed) throw StateError('SessionState已释放');

    _logger.debug('序列化会话状态');

    final data = <String, dynamic>{
      'session_id': sessionId,
      'created_at': createdAt.toIso8601String(),
      'n_past': _cacheSnapshot?.nPast ?? 0,
      'keep_prefix': _cacheSnapshot?.keepPrefix ?? 0,
      'state_data': _cacheSnapshot?.stateData != null ? base64Encode(_cacheSnapshot!.stateData!) : '',
    };

    final jsonStr = jsonEncode(data);
    _serializedData = Uint8List.fromList(jsonStr.codeUnits);

    _logger.debug('序列化为${_serializedData!.length}字节');
    return _serializedData!;
  }

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
      throw LlamaException.session('反序列化失败: $e');
    }
  }

  Future<File> saveToFile(String path) async {
    if (_disposed) throw StateError('SessionState已释放');

    _logger.info('保存会话到文件: $path');

    final data = await serialize();
    final file = File(path);

    final dir = file.parent;
    if (!dir.existsSync()) {
      dir.createSync(recursive: true);
    }

    await file.writeAsBytes(data);

    _logger.info('会话已保存: ${file.path}');
    return file;
  }

  static Future<SessionState> loadFromFile(String path) async {
    final file = File(path);

    if (!file.existsSync()) {
      throw FileSystemException('会话文件未找到', path);
    }

    final data = await file.readAsBytes();
    return deserialize(data);
  }

  int get bufferSize => _serializedData?.length ?? 0;
  bool get hasData => _serializedData != null;

  void clear() {
    _cacheSnapshot = null;
    _serializedData = null;
    _logger.debug('会话数据已清空');
  }

  void dispose() {
    if (_disposed) return;

    _logger.debug('释放SessionState');
    clear();
    _disposed = true;
  }

  bool get isDisposed => _disposed;
}

class SessionManager {
  final Map<String, SessionState> _sessions = {};
  final Logger _logger = Logger('SessionManager');

  SessionState createSession() {
    final session = SessionState();
    _sessions[session.sessionId] = session;
    _logger.debug('创建会话: ${session.sessionId}');
    return session;
  }

  SessionState? getSession(String sessionId) {
    return _sessions[sessionId];
  }

  void removeSession(String sessionId) {
    final session = _sessions.remove(sessionId);
    session?.dispose();
    _logger.debug('移除会话: $sessionId');
  }

  List<String> listSessions() {
    return _sessions.keys.toList();
  }

  void clearAll() {
    for (final session in _sessions.values) {
      session.dispose();
    }
    _sessions.clear();
    _logger.debug('所有会话已清空');
  }

  Future<void> saveAllToDirectory(String dirPath) async {
    final dir = Directory(dirPath);
    if (!dir.existsSync()) {
      dir.createSync(recursive: true);
    }

    for (final session in _sessions.values) {
      final path = '${dirPath}/${session.sessionId}.session';
      await session.saveToFile(path);
    }

    _logger.info('已保存${_sessions.length}个会话');
  }

  Future<void> loadAllFromDirectory(String dirPath) async {
    final dir = Directory(dirPath);
    if (!dir.existsSync()) return;

    final files = dir.listSync().whereType<File>().where((f) => f.path.endsWith('.session'));

    for (final file in files) {
      try {
        final session = await SessionState.loadFromFile(file.path);
        _sessions[session.sessionId] = session;
      } catch (e) {
        _logger.error('加载会话失败: ${file.path}', e);
      }
    }

    _logger.info('已加载${_sessions.length}个会话');
  }
}
