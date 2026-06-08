# ComfyUI 爆款封面 LLM 提示词插件

这个插件只保留一个节点：

`爆款封面LLM提示词`

它会调用 RunningHub 的 LLM 能力，根据用户上传的图片或文字需求，生成一段可以继续接到生图节点使用的中文完整封面提示词。

图片是可选项：接入图片时按图生图/识图封面提示词生成；不接图片时按文生图封面提示词生成。

## 节点输入

- `model`：RunningHub LLM 模型下拉框，优先从 `https://llm.runninghub.ai/v1/models` 获取
- `封面风格`：封面风格预设下拉框
- `主题关键词`：封面围绕的主题，例如 AI提示词、穿搭教程、产品种草、工具介绍等
- `封面标题`：希望出现在封面上的中文标题；可以留空，留空时由 LLM 根据主题自动生成
- `加载图像`：可选，接 ComfyUI 自带的 Load Image 节点
- `api_key`：可选，RunningHub 平台环境通常不需要填写
- `api_baseurl`：默认 `https://llm.runninghub.cn/v1`
- `自定义要求`：可选，用于写自定义风格、标题位置、画面比例、保留元素、不要文字等要求
- `temperature`
- `max_tokens`

## 封面风格

当前内置风格：

- `小红书干净高级风`
- `强冲击爆款标题风`
- `科技感教程封面风`
- `商业海报风`
- `杂志大片风`
- `可爱手账风`
- `电商产品种草风`
- `真实生活方式风`
- `自定义`

如果选择 `自定义`，请在 `自定义要求` 里填写具体风格描述，例如：

```text
韩系杂志感，白色背景，人物偏右，标题放左侧，干净高级
```

## 节点输出

- `系统提示词`

这里的输出实际是 LLM 根据图片或文字需求生成的完整封面提示词，可以继续接到后面的图像生成节点。

## 图片识别

接入图片时，节点会把图片转换成 JPEG base64，并以 OpenAI 兼容的多模态格式发送：

```text
text + image_url
```

运行时后台会打印类似信息：

```json
{
  "has_image": true,
  "content_type": "multimodal",
  "image_size": [1080, 1922]
}
```

如果看到 `has_image: true`，说明节点已经收到图片并随请求发送给模型。

## 内容安全

节点会在提示词生成前后做内容安全收敛，尽量避免后面的生图节点触发内容安全审查。

例如会把容易触发审查的描述替换成更中性的封面描述：

```text
性感 -> 时尚
比基尼 -> 日常服装
低胸 -> 简洁领口
腿部特写 -> 全身构图
```

最终是否通过安全审查仍取决于后续生图模型和平台策略，但节点会尽量避免输出擦边、低俗、过度暴露和身体局部特写类提示词。

## API Key

在 RunningHub 平台环境上通常不需要手动填写 `api_key`。

本地自测或接外部 OpenAI 兼容接口时，如果你的环境需要鉴权，可以在节点里的 `api_key` 输入框填写，或者设置环境变量：

```bash
RH_API_KEY=你的RunningHubKey
```

## RunningHub LLM

节点默认调用 RunningHub 官方 LLM OpenAI 兼容接口：

```text
https://llm.runninghub.cn/v1/chat/completions
```

模型下拉框会优先从 RunningHub 模型接口获取：

```text
https://llm.runninghub.ai/v1/models
```

如果当前运行环境临时拉不到模型接口，节点会显示备用模型列表。

## 在 ComfyUI 里搜什么

搜索：

```text
爆款封面
```

或：

```text
爆款封面LLM提示词
```

## 系统提示词维护

系统提示词在：

```text
system_prompt.txt
```

以后要调整爆款封面规则、识图策略、标题生成规则、输出格式，优先改这个文件。

## 安装

把整个仓库文件夹放进 ComfyUI 的：

```text
custom_nodes/
```

然后重启 ComfyUI。
