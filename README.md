# ComfyUI 爆款封面 LLM 提示词插件

这个插件只保留一个节点：

`爆款封面LLM提示词`

它会读取一张图片，调用 RunningHub 的 LLM 图像理解能力，再根据内置系统提示词输出一段完整封面提示词。

## 节点输入

- `加载图像`：接 ComfyUI 自带的 Load Image 节点
- `api_key`：RunningHub API Key
- `model`：RunningHub LLM 模型
- `平台类型`
- `内容类型`
- `封面风格`
- `主题关键词`
- `封面标题`
- `补充要求`
- `api_baseurl`
- `temperature`
- `max_tokens`

## 节点输出

- `系统提示词`

这里的输出实际是 LLM 根据图片生成的完整封面提示词，可以继续接到后面的图像生成节点。

## 在 ComfyUI 里搜什么

搜索：

`爆款封面`

或：

`爆款封面LLM提示词`

## API Key

优先在节点里的 `api_key` 输入框填写。

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

这个调用方式参考 RunningHub LLM 插件的 OpenAI 兼容接口。

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
