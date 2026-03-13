/// 可释放接口
mixin Disposable {
  /// 释放资源
  void dispose();
  
  /// 是否已释放
  bool get isDisposed;
}
