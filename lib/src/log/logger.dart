import 'package:logger/logger.dart' as logger;

/// 统一日志记录器
class Logger {
  final String _tag;
  final logger.Logger _logger;

  /// 创建日志记录器
  Logger(this._tag, {logger.Level level = logger.Level.info})
      : _logger = logger.Logger(
          printer: logger.PrettyPrinter(
            methodCount: 0,
            errorMethodCount: 2,
            lineLength: 80,
            colors: true,
            printEmojis: false,
            printTime: true,
          ),
          level: level,
        );

  /// 调试日志
  void debug(String message) {
    _logger.d('[$_tag] $message');
  }

  /// 信息日志
  void info(String message) {
    _logger.i('[$_tag] $message');
  }

  /// 警告日志
  void warning(String message) {
    _logger.w('[$_tag] $message');
  }

  /// 错误日志
  void error(String message, [dynamic error, StackTrace? stackTrace]) {
    _logger.e('[$_tag] $message', error: error, stackTrace: stackTrace);
  }
}
