# Qwen-Audio 云端文件转录

在转录模型中选择 **Qwen-Audio 云端 ASR [API] ✨**，填写千问 API Key。默认基础地址为 `https://maas.qianwenaiapi.com/api/v1`，默认模型为 `qwen-audio-3.1-asr-flash-filetrans`，对应[千问模型页面](https://www.qianwenai.com/models/qwen-audio-3.1-asr-flash-filetrans)。这项云端服务独立于已有的本地 Qwen3-ASR 和 MiMo-ASR，无需安装本地识别、VAD 或对齐模型。

选择本地视频或音频后，程序提取音频，获取临时 OSS 上传凭证，上传文件，再提交异步识别任务。文件上传后有效期为 48 小时；开始转录会上传音频并产生服务商识别费用。“测试连接”只获取上传凭证，验证 API Key 和模型权限，不上传音频或提交识别任务。该测试通过后，实际识别仍可能受配额、余额或模型权限限制。

## 热词与上下文增强

在“自定义热词”中每行填写一个词条，默认权重为 4。需要指定权重时写为 `词条 | 权重`：

```text
斯蒂格勒
第三持存 | 5
VideoCaptioner | 4
```

权重必须为整数 1–5 或 50；50 是超级热词，适合少量容易误识别的关键术语。总计最多 2000 条，其中超级热词最多 50 条。含中文、重音字母等非 ASCII 字符的词条最多 15 个字符；纯 ASCII 词条最多 7 个以空白分隔的单词。重复词条、超限或错误权重会在界面显示具体行号，并在上传前拒绝提交，程序不会截断词表。

“已有云端词表 ID”可填入服务商预先创建的词表 ID；其内容与上方热词合并使用。程序不创建、编辑或查询云端词表。由于无法在本地统计云端词表内容，合并总量超过 2000 条时，服务可能只选用部分热词；请自行控制两个列表的合计数量。填写云端词表 ID 后程序不读写转录缓存，避免同一 ID 的内容更新后复用旧结果。

“上下文增强”最多填写 400 个字符。说明录音的主题，列出人名与术语的准确写法，例如“录音讨论斯蒂格勒的第三持存，术语包括 rétention tertiaire 和 néguanthropie”。保留音频中的原文词形，使用简短、相关的信息。留空时不发送上下文。热词与上下文会发送给识别服务；界面中的内容保存在本地，任务配置日志不记录这些正文。

热词和上下文的编辑区会保留光标与撤销记录，并在转录页和全局设置之间同步。转录页上方的命令栏保持可见，下方媒体信息与设置表单可以滚动；在较小窗口中向下滚动即可使用热词、上下文以及最后的测试连接按钮。

基础地址须对应文件转录 REST API，支持 `/uploads`、`/services/audio/asr/transcription` 和 `/tasks/{task_id}`；OpenAI 兼容聊天地址不适用于此服务。使用自定义渠道时，渠道也须支持临时 OSS 上传。程序使用指定模型，不会自动切换模型。

程序直接保留服务返回的毫秒时间戳。独立转录默认生成句子字幕；开启“词级时间戳”后使用服务返回的词时间戳，保留标点与空格。某个句子没有词级结果时保留该句的原生时间戳；有文本却缺少有效时间戳会报告错误。完整处理流程需要字幕断句时可自动使用词级结果。

单个音频须不超过 2 GB、12 小时，且不能超过当前上传凭证规定的文件大小。程序在上传前检查大小与时长，流式上传文件，不将整个长音频载入内存。读取非 WAV 音频时长和提取视频音轨需要 FFmpeg/ffprobe。

识别默认最多等待 30 分钟，状态每 3 秒查询一次。查询与下载的临时网络错误最多尝试 3 次，并遵守服务返回的 `Retry-After`；任务提交只在明确收到 HTTP 429 拒绝后重试。提交超时不自动重复提交，以免产生重复识别费用。取消会停止本地上传、查询和字幕保存；已经提交的云端任务可能继续执行并计费。网络请求中的取消最长可能需要等待当前请求超时。

## 命令行

建议通过环境变量提供密钥：

```powershell
$env:VIDEOCAPTIONER_QWEN_FILETRANS_API_KEY = "你的千问 API Key"
videocaptioner transcribe "视频.mp4" --asr qwen-filetrans -o "字幕.srt"
```

语言提示和词级字幕示例：

```powershell
videocaptioner transcribe "音频.wav" --asr qwen-filetrans --language zh --word-timestamps
```

`process` 命令也支持 `--asr qwen-filetrans`。可选参数为 `--qwen-filetrans-key`、`--qwen-filetrans-base`、`--qwen-filetrans-model`。环境变量还支持 `VIDEOCAPTIONER_QWEN_FILETRANS_API_BASE` 和 `VIDEOCAPTIONER_QWEN_FILETRANS_MODEL`。

使用 UTF-8 热词文件与上下文：

```powershell
videocaptioner transcribe "录音.wav" --asr qwen-filetrans --qwen-filetrans-hotwords-file "热词.txt" --qwen-filetrans-context "录音讨论斯蒂格勒的第三持存。"
```

热词文件可带 UTF-8 BOM，格式与界面相同。也可用 `--qwen-filetrans-hotwords` 直接传入多行文本，以及 `--qwen-filetrans-vocabulary-id` 指定已有云端词表。`transcribe` 和 `process` 均支持这些参数。

配置文件示例：

```toml
[transcribe]
asr = "qwen-filetrans"
language = "auto"

[qwen_filetrans]
api_key = ""
api_base = "https://maas.qianwenaiapi.com/api/v1"
model = "qwen-audio-3.1-asr-flash-filetrans"
task_timeout = 1800
hotwords = "斯蒂格勒\n第三持存 | 5"
vocabulary_id = ""
context = "录音讨论斯蒂格勒的第三持存。"
```

`task_timeout` 以秒为单位，可用 `videocaptioner config set qwen_filetrans.task_timeout 3600` 增加等待时间。支持的语言提示包含中文、英语、日语、韩语等 30 种语言；界面只列出服务支持的语言提示。方言建议使用自动识别或中文提示。

参考官方[准确率增强指南](https://platform.qianwenai.com/docs/developer-guides/speech/improve-recognition-accuracy)、[文件转录协议](https://platform.qianwenai.com/docs/developer-guides/speech/asr)、[临时上传接口](https://platform.qianwenai.com/docs/api-reference/more/upload-file-get-temporary-url)和[文件识别 HTTP 参数与结果格式](https://help.aliyun.com/zh/model-studio/fun-asr-recorded-speech-recognition-http-api)。文件转录使用 `input.context` 消息数组和 `parameters.vocabulary` 词到权重的映射；同步模型示例中的 `input.messages` 不适用于这里。当前验证覆盖模拟接口、实际 HTTP 请求序列化、参数边界、配置及缓存行为，以及完整转录界面的滚动、编辑与切换；尚未使用真实账户进行收费端到端转录。
