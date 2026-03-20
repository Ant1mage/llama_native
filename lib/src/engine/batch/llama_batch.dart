import 'package:llama_native/src/log/logger.dart';

class LlamaBatch {
  final Logger _logger = Logger('LlamaBatch');

  final List<int> tokens;
  final bool logitsAll;

  LlamaBatch(this.tokens, {this.logitsAll = false});

  LlamaBatch.single(this.tokens, {this.logitsAll = false});

  void dispose() {
    _logger.debug('Batch已释放');
  }
}
