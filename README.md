# ComfyUI 爆款封面 LLM 提示词插件

这个插件只保留一个节点：

`爆款封面LLM提示词`

它会调用 RunningHub 的 LLM 能力，再根据内置系统提示词输出一段完整封面提示词。

图片是可选项：接入图片时按图生图/识图提示词生成；不接图片时按文生图提示词生成。

## 节点输入

- `model`：RunningHub LLM 模型
- `平台类型`
- `内容类型`
- `封面风格`
- `主题关键词`
- `封面标题`
- `加载图像`：可选，接 ComfyUI 自带的 Load Image 节点
- `api_key`：可选，本地自测或外部接口需要时再填
- `补充要求`
- `api_baseurl`
- `api_config`
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

## API Key

在 RunningHub 平台上通常不需要手动填写 `api_key`。

本地自测或接外部 OpenAI 兼容接口时，可以在节点里的 `api_key` 输入框填写。

也可以设置环境变量：

```bash
RH_API_KEY=你的RunningHubKey
```

节点默认调用：

```text
https://llm.runninghub.cn/v1
```

节点会自动调用：

```text
{api_baseurl}/chat/completions
```

这个调用方式参考 RunningHub 官方 `RH LLM Chat Completions` 节点的 OpenAI 兼容接口。

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
