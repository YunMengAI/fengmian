# ComfyUI 爆款封面 LLM 提示词插件

这个插件只保留一个节点：

`爆款封面LLM提示词`

它会调用 RunningHub 的 LLM 能力，再根据内置系统提示词输出一段完整封面提示词。

图片是可选项：接入图片时按图生图/识图提示词生成；不接图片时按文生图提示词生成。

## 节点输入

- `model`：RunningHub LLM 模型下拉框，优先从 `https://llm.runninghub.ai/v1/models` 获取
- `平台类型`
- `内容类型`
- `封面风格`
- `主题关键词`
- `封面标题`
- `加载图像`：可选，接 ComfyUI 自带的 Load Image 节点
- `api_key`：可选，在 RunningHub 平台上通常不用填
- `补充要求`
- `temperature`
- `max_tokens`

## 节点输出

- `系统提示词`

这里的输出实际是 LLM 根据图片或文字需求生成的完整封面提示词，可以继续接到后面的图像生成节点。

## 在 ComfyUI 里搜什么

搜索：

`爆款封面`

或：

`爆款封面LLM提示词`

## RunningHub LLM

节点使用 RunningHub 官方 LLM OpenAI 兼容接口：

```text
https://llm.runninghub.cn/v1/chat/completions
```

模型下拉框会优先从 RunningHub 模型接口获取：

```text
https://llm.runninghub.ai/v1/models
```

如果当前运行环境临时拉不到模型接口，节点会显示 RunningHub 官方节点同款备用模型列表。

## API Key

在 RunningHub 平台上通常不需要手动填写 `api_key`，平台环境可以使用 RH 自己的 LLM 能力。

本地自测时，如果你的环境需要鉴权，可以填写节点里的 `api_key`，或者设置环境变量：

```bash
RH_API_KEY=你的RunningHubKey
```

## 系统提示词维护

系统提示词在：

`system_prompt.txt`

以后你要调整爆款封面规则、识图策略、输出格式，优先改这个文件，不需要改 Python 代码。

## 安装

把整个仓库文件夹放进 ComfyUI 的：

```text
custom_nodes/
```

然后重启 ComfyUI。
