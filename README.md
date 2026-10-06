<div align="center">
  <img src="./docs/images/logo.png" alt="VideoCaptioner Logo" width="100">
  <h1>VideoCaptioner</h1>
  <p>基于大语言模型的视频字幕处理工具 — 语音识别、字幕优化、翻译、视频合成一站式处理</p>

  [在线文档](https://weifeng2333.github.io/VideoCaptioner/) · [CLI 使用](#cli-命令行) · [GUI 桌面版](#gui-桌面版) · [Claude Code Skill](#claude-code-skill)
</div>

## 支持 MiMo 与 Qwen 在线 API

本 fork 的 Windows 定制版同时支持 **MiMo-ASR 在线 API** 和 **Qwen-Audio 云端 ASR API**，并保留 **Qwen3-ASR 本地识别** 与 **FireRedVAD**。两类在线 API 的依赖与运行方式不同：

- **Qwen-Audio 云端 ASR [API]（本次重点新增）**：默认模型为 `qwen-audio-3.1-asr-flash-filetrans`，默认基础地址为 `https://maas.qianwenaiapi.com/api/v1`。程序自动提取本地音视频音轨并流式上传至千问临时 OSS，异步转写并获取原生句子或词级时间戳（毫秒单位）。**无需单独配置对象存储，无需下载本地识别、VAD 或对齐模型**（视频提取音频需 FFmpeg）。支持自定义热词（支持权重）与上下文增强（最多 400 字符）。在线识别需要服务商 API Key，并按服务商规则计费（“测试连接”仅验证上传权限，不提交收费识别）。
- **MiMo-ASR 在线 API（既有支持）**：可配置渠道地址、API Key 和模型。短字幕生成依赖本地共享运行环境与 `Qwen3-ForcedAligner-0.6B` 对齐模型；VAD 默认开启但可关闭，仅在开启时需要本地 VAD 组件（支持 Silero / FireRedVAD）。已有组件可直接复用，无需下载 Qwen3-ASR 识别权重。

**Windows 定制版下载**：前往 [Releases 最新发布页面](https://github.com/kawainotu/VideoCaptioner-qwen3asr-fireredvad/releases/latest) 获取安装包 `VideoCaptioner-Setup-win64-v1.4.2-qwen3-asr-fireredvad-1.4.exe`。

**详细配置指南**：参见 [Qwen-Audio 云端配置](docs/config/qwen-filetrans.md) 与 [MiMo-ASR 配置说明](docs/config/asr.md#小米-mimo-asr-api-配置说明)。

## 安装

> [!IMPORTANT]
> **注意**：下方 `pip install videocaptioner` 命令安装的是**上游官方发布版**。若要使用本 fork 提供的 Windows 定制功能（Qwen 云端文件转录、MiMo 在线 API、Qwen3-ASR 本地识别与 FireRedVAD 等），请直接下载 [Windows 定制版安装包 (Releases)](https://github.com/kawainotu/VideoCaptioner-qwen3asr-fireredvad/releases/latest)，或从[本仓库源码](#开发)在本地运行。

```bash
pip install videocaptioner          # 仅安装上游官方 CLI + GUI 版本
```

免费功能（必剪语音识别、必应/谷歌翻译）**无需任何配置，安装即用**。

## 本 fork 的改动

此 fork 提供面向 Windows 的本地与在线语音识别定制安装包：

- **新增 Qwen-Audio 在线文件转录 API**：支持异步上传、原生句级/词级时间戳（毫秒单位）、自定义热词（含权重）与上下文增强，无需本地识别、VAD 或对齐模型，GUI 与 CLI 均可使用。
- **转录页面滚动与交互修复**：固定顶部操作栏，媒体信息与完整设置表单自适应滚动，小窗口下可正常配置并使用“测试连接”；热词与上下文在语音转录页面的设置表单中编辑，保留光标和撤销历史，并与全局设置实时双向同步。
- **保留 MiMo-ASR 在线 API 与修复**：支持自定义服务渠道、频率调度与本地字幕对齐；长音频中“好。”“嗯。”等严格受限短应答回退使用已检测的 VAD 句级边界并标记需复核，避免整段字幕导出失败。
- **保留 Qwen3-ASR 本地识别**：支持 CUDA、CPU 和自动设备选择，提供官方轻量 `0.6B` 与 `1.7B` 模型管理。
- **保留 FireRedVAD**：内置离线权重，支持 Silero VAD 与 FireRedVAD 切换及切分参数调优。
- **组件管理与安装包打包**：Windows 安装包内置 Qwen3-ASR 运行程序与 FireRedVAD 权重，支持从国内镜像下载环境与模型。

Windows 定制版安装包与更新说明见 [本 fork 的 Releases](https://github.com/kawainotu/VideoCaptioner-qwen3asr-fireredvad/releases)。首次使用 Qwen3-ASR 前，需要在“管理组件”下载模型；模型文件约 6.5 GB。

## CLI 命令行

```bash
# 语音转录（免费，无需 API Key）
videocaptioner transcribe video.mp4 --asr bijian

# Qwen-Audio 云端识别（需在配置中填写 qwen_filetrans.api_key）
videocaptioner transcribe video.mp4 --asr qwen-filetrans

# 字幕翻译（免费必应翻译）
videocaptioner subtitle input.srt --translator bing --target-language en

# 全流程：转录 → 优化 → 翻译 → 合成
videocaptioner process video.mp4 --target-language ja

# 字幕烧录到视频
videocaptioner synthesize video.mp4 -s subtitle.srt

# 下载在线视频
videocaptioner download "https://youtube.com/watch?v=xxx"
```

需要 LLM 功能（字幕优化、大模型翻译）时，配置 API Key：

```bash
videocaptioner config set llm.api_key <your-key>
videocaptioner config set llm.api_base https://api.openai.com/v1
videocaptioner config set llm.model gpt-4o-mini
```

配置优先级：`命令行参数 > 环境变量 (VIDEOCAPTIONER_*) > 配置文件 > 默认值`。运行 `videocaptioner config show` 查看当前配置。

<details>
<summary>所有 CLI 命令一览</summary>

| 命令 | 说明 |
|------|------|
| `gui` | 打开桌面版。也可以直接运行 `videocaptioner-gui` |
| `transcribe` | 语音转字幕。引擎：`qwen-filetrans`（Qwen-Audio 在线 API）、`qwen3-asr`（Windows 本地）、`faster-whisper`、`whisper-api`、`bijian`（免费）、`jianying`（免费）、`whisper-cpp` |
| `subtitle` | 字幕优化/翻译。翻译服务：`llm`、`bing`（免费）、`google`（免费） |
| `dub` | 根据字幕生成配音音轨或配音视频 |
| `synthesize` | 字幕烧录到视频（软字幕/硬字幕） |
| `process` | 全流程处理 |
| `download` | 下载 YouTube、B站等平台视频 |
| `config` | 配置管理（`show`、`set`、`get`、`path`、`init`） |

运行 `videocaptioner <命令> --help` 查看完整参数。完整 CLI 文档见 [docs/cli.md](docs/cli.md)。

</details>

## GUI 桌面版

```bash
pip install videocaptioner
videocaptioner-gui                  # 显式打开桌面版
videocaptioner gui                  # 等价命令
videocaptioner                      # 无参数时也会打开桌面版
```

<details>
<summary>其他安装方式：Windows 安装包 / macOS 一键脚本</summary>

**Windows**：从 [本 fork 的 Releases](https://github.com/kawainotu/VideoCaptioner-qwen3asr-fireredvad/releases) 下载安装包

**macOS**：
```bash
curl -fsSL https://raw.githubusercontent.com/WEIFENG2333/VideoCaptioner/master/scripts/run.sh | bash
```

</details>


<!-- <div align="center">
  <img src="https://h1.appinn.me/file/1731487405884_main.png" alt="界面预览" width="90%" style="border-radius: 5px;">
</div> -->

![页面预览](https://h1.appinn.me/file/1731487410170_preview1.png)
![页面预览](https://h1.appinn.me/file/1731487410832_preview2.png)

### Qwen3-ASR 本地识别（Windows）

在转录模型中选择 `Qwen3-ASR`，然后打开“管理组件”安装独立运行环境、识别模型与 `Qwen3-ForcedAligner-0.6B`。可在“Qwen3 系列模型”中选择官方 `Qwen3-ASR-0.6B`（约 1.88 GB）或 `Qwen3-ASR-1.7B`；前者与时间戳模型合计约 3.7 GB，适合更关注显存和下载体积的场景。

Qwen3-ASR 的 Python、PyTorch、Silero VAD、FireRedVAD 运行环境以及时间戳模型由所有 Qwen3-ASR 识别模型共用，只需安装一次；切换 0.6B、1.7B 或微调模型时不会重复下载 PyTorch。安装器默认使用阿里云 PyPI/PyTorch 镜像，Hugging Face 模型默认使用 `hf-mirror.com`，镜像失败时会自动回退官方源。可通过 `VIDEOCAPTIONER_PYPI_INDEX`、`VIDEOCAPTIONER_PYTORCH_MIRROR` 和 `VIDEOCAPTIONER_HF_ENDPOINT` 覆盖下载源。

默认开启低显存模式，识别与时间戳对齐会分阶段加载模型，适合 8 GB 显存设备。Qwen3-ASR 设置页提供 CUDA/CPU/自动设备选择、Silero VAD / FireRedVAD 切换、对应的阈值与切分参数，以及上下文提示。发布安装包会内置 FireRedVAD 离线权重，也可在“管理组件”中更新。

CLI 使用示例：

```bash
# 使用官方轻量 0.6B 模型（首次使用前请在 GUI 的“管理组件”中下载）
videocaptioner config set transcribe.qwen3_asr.model qwen3-asr-0.6b
videocaptioner transcribe video.mp4 --asr qwen3-asr
```

## LLM API 配置

LLM 仅用于字幕优化和大模型翻译，免费功能（必剪识别、必应翻译）无需配置。

支持所有 OpenAI 兼容接口的服务商：

| 服务商 | 官网 |
|--------|------|
| **VideoCaptioner 中转站** | [api.videocaptioner.cn](https://api.videocaptioner.cn) — 高并发，性价比高，支持 GPT/Claude/Gemini 等 |
| SiliconCloud | [cloud.siliconflow.cn](https://cloud.siliconflow.cn/i/HF95kaoz) |
| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com) |

在软件设置或 CLI 中填入 API Base URL 和 API Key 即可。[详细配置教程](https://weifeng2333.github.io/VideoCaptioner/config/llm)

## Claude Code Skill

本项目提供了 [Claude Code Skill](https://code.claude.com/docs/en/skills.md)，让 AI 编程助手可以直接调用 VideoCaptioner 处理视频。

安装到 Claude Code：

```bash
mkdir -p ~/.claude/skills/videocaptioner
cp skills/SKILL.md ~/.claude/skills/videocaptioner/SKILL.md
```

然后在 Claude Code 中输入 `/videocaptioner transcribe video.mp4 --asr bijian` 即可使用。

## 工作原理

```
音视频输入 → 语音识别 → 字幕断句 → LLM 优化 → 翻译 → 视频合成
```

- 词级时间戳 + VAD 语音活动检测，识别准确率高
- LLM 语义理解断句，字幕阅读体验自然流畅
- 上下文感知翻译，支持反思优化机制
- 批量并发处理，效率高

## 开发

```bash
git clone https://github.com/kawainotu/VideoCaptioner-qwen3asr-fireredvad.git
cd VideoCaptioner-qwen3asr-fireredvad
uv sync && uv run videocaptioner     # 运行 GUI
uv run videocaptioner --help          # 运行 CLI
uv run pyright                        # 类型检查
uv run pytest tests/test_cli/ -q      # 运行测试
```

## 许可证

[GPL-3.0](LICENSE)

[![Star History Chart](https://api.star-history.com/svg?repos=WEIFENG2333/VideoCaptioner&type=Date)](https://star-history.com/#WEIFENG2333/VideoCaptioner&Date)
