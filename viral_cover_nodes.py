import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path


RUNNINGHUB_LLM_CHAT_URL = "https://llm.runninghub.cn/v1/chat/completions"
RUNNINGHUB_LLM_MODELS_URL = "https://llm.runninghub.ai/v1/models"
RUNNINGHUB_DEFAULT_MODELS = [
    "google/gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview",
    "google/gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash-lite-preview",
]
RUNNINGHUB_FALLBACK_MODELS = [
    "doubao-seed-1-6-251015",
    "doubao-seed-2-0-pro-260215",
    "doubao-seed-2-0-lite-260215",
    "DeepSeek-V3",
    "Qwen3-235B-A22B-Instruct-2507",
    "gemini-2.5-flash",
    "gpt-5",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "GLM-4.6v",
]

PLATFORMS = [
    "小红书封面",
    "抖音视频封面",
    "快手视频封面",
    "通用内容封面",
    "课程封面",
]

CONTENT_TYPES = [
    "AI教程",
    "穿搭教程",
    "美妆护肤",
    "知识干货",
    "科技科普",
    "产品种草",
    "爆款标题封面",
]

STYLES = [
    "小红书ins风",
    "强冲击卖点风",
    "杂志封面风",
    "赛博科技风",
    "可爱手账风",
    "商业海报风",
]

DEFAULT_BASE_URL = RUNNINGHUB_LLM_CHAT_URL
SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.txt")


def load_system_prompt():
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def fetch_runninghub_models():
    try:
        request = urllib.request.Request(RUNNINGHUB_LLM_MODELS_URL, method="GET")
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read().decode("utf-8", errors="replace")
        data = json.loads(body)
        models = [
            str(item.get("id", "")).strip()
            for item in data.get("data", [])
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        ]
        if models:
            return models
    except Exception as exc:
        print(f"[ViralCoverLLMPrompt] Failed to fetch RunningHub models, using fallback: {type(exc).__name__}")

    return list(RUNNINGHUB_FALLBACK_MODELS)


def default_runninghub_model(models):
    for model in RUNNINGHUB_DEFAULT_MODELS:
        if model in models:
            return model
    return models[0]


def get_config_value(config, *keys):
    if not isinstance(config, dict):
        return ""

    for key in keys:
        value = str(config.get(key, "")).strip()
        if value:
            return value
    return ""


def get_api_key(api_key=None, api_config=None, llm_config=None):
    for config in (api_config, llm_config):
        value = get_config_value(config, "api_key", "apiKey", "apikey")
        if value:
            return value

    api_key = (api_key or "").strip()
    if api_key:
        return api_key

    env_key = (os.environ.get("RUNNINGHUB_LLM_API_KEY") or os.environ.get("RH_LLM_API_KEY") or "").strip()
    if env_key:
        return env_key

    try:
        from server import PromptServer

        shared_key = getattr(PromptServer.instance, "shared_api_key", None)
        if isinstance(shared_key, str) and shared_key.strip() and shared_key != "unknown":
            return shared_key.strip()
    except Exception:
        pass

    env_key = (os.environ.get("RH_API_KEY") or os.environ.get("RUNNINGHUB_API_KEY") or "").strip()
    if env_key:
        return env_key

    return ""


def resolve_base_url(api_baseurl="", api_config=None, llm_config=None):
    for config in (api_config, llm_config):
        value = get_config_value(config, "base_url", "api_url", "api_baseurl")
        if value:
            return value
    return (api_baseurl or DEFAULT_BASE_URL).strip()


def resolve_model(model, api_config=None, llm_config=None):
    for config in (api_config, llm_config):
        value = get_config_value(config, "model_name", "models_name", "model")
        if value:
            return value
    return model


def normalize_chat_endpoint(base_url):
    base_url = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def tensor_to_jpeg_data_url(image):
    from PIL import Image

    array = image
    if hasattr(array, "detach"):
        if array.ndim == 4:
            array = array[0]
        array = array.mul(255.0).clamp(0, 255).byte().cpu().numpy()
    else:
        import numpy as np

        if array.ndim == 4:
            array = array[0]
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(array)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def remove_think_tags(text):
    if not isinstance(text, str):
        return text

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|begin_of_box\|>|<\|end_of_box\|>", "", text)
    return text.strip()


def build_user_prompt(平台类型, 内容类型, 封面风格, 主题关键词, 封面标题, 补充要求):
    return f"""请根据我提供的图片或文字需求，结合以下变量，输出一段可直接用于 AI 绘图的完整封面提示词。

平台类型：{平台类型}
内容类型：{内容类型}
封面风格：{封面风格}
主题关键词：{主题关键词}
封面标题：{封面标题}
补充要求：{补充要求.strip() or "无"}

如果我提供了图片，请重点分析原图里的主体、构图、色彩、光影、背景、人物/产品状态、可复用的视觉卖点。
如果没有图片，请直接根据文字变量生成适合文生图的封面提示词。
输出时只给最终提示词，不要解释过程。"""


def post_json(endpoint, headers, payload):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body


class ViralCoverLLMPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        models = fetch_runninghub_models()
        return {
            "required": {
                "model": (models, {"default": default_runninghub_model(models)}),
                "平台类型": (PLATFORMS,),
                "内容类型": (CONTENT_TYPES,),
                "封面风格": (STYLES,),
                "主题关键词": ("STRING", {"default": "7天学会AI绘画", "multiline": False}),
                "封面标题": ("STRING", {"default": "新手也能做出爆款封面", "multiline": False}),
            },
            "optional": {
                "加载图像": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "补充要求": ("STRING", {"default": "", "multiline": True}),
                "api_baseurl": ("STRING", {"default": DEFAULT_BASE_URL, "multiline": False}),
                "api_config": ("RH_OPENAPI_CONFIG",),
                "llm_config": ("SYNVOW_LLM_CONFIG",),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("系统提示词",)
    FUNCTION = "generate"
    CATEGORY = "爆款封面生成"

    def generate(
        self,
        model,
        平台类型,
        内容类型,
        封面风格,
        主题关键词,
        封面标题,
        加载图像=None,
        api_key="",
        补充要求="",
        api_baseurl=DEFAULT_BASE_URL,
        api_config=None,
        llm_config=None,
        temperature=0.4,
        max_tokens=2048,
    ):
        role = load_system_prompt()
        prompt = build_user_prompt(平台类型, 内容类型, 封面风格, 主题关键词, 封面标题, 补充要求)
        model = resolve_model(model, api_config, llm_config)
        endpoint = normalize_chat_endpoint(resolve_base_url(api_baseurl, api_config, llm_config))
        headers = {
            "Content-Type": "application/json",
        }
        api_key_value = get_api_key(api_key, api_config, llm_config)
        if api_key_value:
            headers["Authorization"] = f"Bearer {api_key_value}"

        user_content = prompt
        if 加载图像 is not None:
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": tensor_to_jpeg_data_url(加载图像)}},
            ]

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": role},
                {"role": "user", "content": user_content},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "reasoning_effort": "none",
        }

        try:
            status_code, body = post_json(endpoint, headers, payload)
            data = json.loads(body)
        except ValueError as exc:
            raise RuntimeError(f"RunningHub LLM 返回了非 JSON 内容：{body[:200]}") from exc

        if status_code != 200:
            message = data.get("error") or data.get("message") or data.get("msg") or body[:200]
            raise RuntimeError(f"RunningHub LLM 调用失败：HTTP {status_code}: {message}")

        choices = data.get("choices")
        if not choices:
            raise RuntimeError("RunningHub LLM 没有返回可用内容。")

        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content") or choices[0].get("text")
        if isinstance(content, list):
            content = "\n".join(
                item.get("text", "") for item in content if isinstance(item, dict) and item.get("text")
            )

        if not content:
            raise RuntimeError("RunningHub LLM 返回内容为空。")

        result = remove_think_tags(str(content))
        print(f"[ViralCoverLLMPrompt] RH LLM request finished at {int(time.time())}")
        print(json.dumps({"model": model, "output_length": len(result)}, ensure_ascii=False))
        return (result,)


NODE_CLASS_MAPPINGS = {
    "ViralCoverLLMPrompt": ViralCoverLLMPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ViralCoverLLMPrompt": "爆款封面LLM提示词",
}
