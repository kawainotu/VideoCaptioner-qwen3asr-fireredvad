# VideoCaptioner mimo 启动脚本修复与诊断报告

## 1. 故障诊断与根本原因 (Root Cause Diagnosis)

- **现象**：用户运行 `launch_mimo_test.bat` 出现闪退、无法启动，且未生成新的 `app.log`。
- **根本原因**：
  1. 原 `launch_mimo_test.bat` 文件采用 **UTF-8 编码**（含中文字符）且为 **纯 LF 换行**（0 个 CRLF，31 个 LF）。
  2. Windows 原生 `cmd.exe` 的批处理解析器依赖 CRLF 寻址。遇到非 ASCII 多字节字符与 UNIX LF 换行组合时，`cmd.exe` 的内部读取缓冲区指针发生严重错位，导致脚本行被错乱截断。
  3. 通过使用非 GUI 的 sentinel stub 测试，真实复现了如下 `cmd.exe` 语法崩溃输出：
     ```text
     '释器是否存在' is not recognized as an internal or external command...
     '项目根目录下创建并安装虚拟环境依赖，或运行' is not recognized...
     'ON_EXE' is not recognized...
     'deocaptioner.ui.main' is not recognized...
     ```
  4. 因此，原脚本在进入 Python 启动命令前即已解析错乱崩溃，Python 解释器从未被实际调用，故不会产生任何 Python 日志。

---

## 2. 修复措施 (Robust Fixes)

针对 Windows 批处理的兼容性要求，对 [launch_mimo_test.bat](file:///C:/Users/kawai/Desktop/VideoCaptioner-1.4.2/launch_mimo_test.bat) 进行了以下稳健修复：

1. **纯 ASCII 字符 (ASCII-only Messages)**：
   - 移除批处理内所有的中文与多字节字符，所有提示、日志和注释全部采用纯 ASCII 英文，从根本上杜绝 Windows 控制台代码页与多字节解析冲突。
2. **显式 CRLF 换行 (Strict CRLF Line Endings)**：
   - 将换行格式规范为 Windows 原生标准 `\r\n`。
3. **安全路径引用 (Quoted Paths)**：
   - 所有变量定义（`set "VAR=value"`）与路径引用（`"%PYTHON_EXE%"`, `"%LOG_FILE%"`, `"%SCRIPT_DIR%"` 等）统一严格加双引号，全面支持含空格路径。
4. **工作目录保持与隔离 (Preserve Working Directory)**：
   - 脚本开头使用 `pushd "%SCRIPT_DIR%"` 确保执行上下文定位在项目根目录，在所有退出路径统一调用 `popd`，不污染调用者的当前工作目录。
5. **错误可见性保障 (Pause on Failure)**：
   - 扁平化错误处理分支（`:on_launch_failure`, `:missing_python`, `:on_check_failure`），在任何非零退出或环境缺失时打印日志详情并执行 `pause`，防止窗口自动关闭闪退。
6. **启动日志捕获 (Capture stdout/stderr to Log)**：
   - 将 Python 进程输出重定向至工作目录下的 [mimo-startup.log](file:///C:/Users/kawai/Desktop/VideoCaptioner-1.4.2/mimo-startup.log)。
   - 发生错误时，自动通过 `type "%LOG_FILE%"` 回显完整的异常堆栈。
7. **提供 `--check` 非 GUI 检查模式**：
   - 支持传入 `--check` 参数执行静默依赖与模块验证（`from videocaptioner.ui.view.main_window import MainWindow`），成功时不暂停直接退出 0，未启动任何 QApplication 或 GUI 界面。

---

## 3. 验证证据 (Verification Evidence)

所有验证均严格遵守非 GUI 原则（无 QApplication、无 main()、无窗口弹出），测试结果如下：

### 3.1 跨目录 `--check` 验证
- **测试命令**：从外部工作目录 `C:\Users\kawai` 与 `C:\` 执行：
  ```cmd
  cmd.exe /c "C:\Users\kawai\Desktop\VideoCaptioner-1.4.2\launch_mimo_test.bat --check"
  ```
- **输出结果**：
  ```text
  [INFO] Running non-GUI dependency and import check...
  [INFO] Interpreter: "C:\Users\kawai\Desktop\VideoCaptioner-1.4.2\.venv\Scripts\python.exe"
  [CHECK-OK] Non-GUI import test succeeded: MainWindow and core dependencies loaded.
  [INFO] Non-GUI check passed successfully.
  ```
- **退出状态**：退出代码 `0`，成功生成 [mimo-startup.log](file:///C:/Users/kawai/Desktop/VideoCaptioner-1.4.2/mimo-startup.log)，无需人工按键交互。

### 3.2 正常分发成功与失败流程验证 (Temp Non-GUI Stub)
- **成功分支**：退出代码 `0`，日志记录正常，正常退出无 pause。
- **失败分支**：模拟 Python 报错（退出码 42），批处理成功捕获非零退出码，完整回显 `mimo-startup.log` 内容并保持 `pause` 等待确认。
- **解释器缺失分支**：模拟 Python 解释器不存在，正常输出 `Python interpreter not found` 并 `pause`。
