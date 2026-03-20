import 'package:logging/logging.dart' as logging;

class Logger {
  final String _tag;
  late final logging.Logger _logger;

  Logger(this._tag) {
    _logger = logging.Logger(_tag);
  }

  static void init({logging.Level level = logging.Level.ALL, bool printTime = true, bool printEmojis = true}) {
    logging.hierarchicalLoggingEnabled = true;
    logging.Logger.root.level = level;

    logging.Logger.root.onRecord.listen((record) {
      final time = printTime ? '${record.time} ' : '';
      final emoji = printEmojis ? _getEmoji(record.level) : '';
      final levelName = record.level.name.padRight(7);

      print('$time$emoji$levelName: ${record.message}');

      if (record.error != null) {
        print('  Error: ${record.error}');
      }
      if (record.stackTrace != null) {
        print('  StackTrace: ${record.stackTrace}');
      }
    });
  }

  static String _getEmoji(logging.Level level) {
    switch (level) {
      case logging.Level.FINE:
      case logging.Level.FINER:
      case logging.Level.FINEST:
        return '🔍 ';
      case logging.Level.INFO:
        return 'ℹ️ ';
      case logging.Level.WARNING:
        return '⚠️ ';
      case logging.Level.SEVERE:
        return '❌ ';
      case logging.Level.SHOUT:
        return '📢 ';
      default:
        return '';
    }
  }

  static Logger getLogger(String tag) {
    return Logger(tag);
  }

  void debug(String message) {
    _logger.fine('[$_tag] $message');
  }

  void info(String message) {
    _logger.info('[$_tag] $message');
  }

  void warning(String message) {
    _logger.warning('[$_tag] $message');
  }

  void error(String message, [dynamic error, StackTrace? stackTrace]) {
    _logger.severe('[$_tag] $message', error, stackTrace);
  }
}
