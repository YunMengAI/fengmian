import base64
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None


LLM_MODELS_URL = "https://llm.runninghub.ai/v1/models"
DEFAULT_API_BASEURL = "https://llm.runninghub.cn/v1"
MODEL_CACHE_TTL_SECONDS = 3600
CHAT_MAX_RETRIES = 3
DEFAULT_MODEL = "google/gemini-3.5-flash"
MODEL_CACHE = {"expires_at": 0.0, "models": None}

FALLBACK_MODELS = [
    DEFAULT_MODEL,
    "qwen/qwen3-vl-235b-a22b-instruct",
    "qwen/qwen-plus",
    "qwen/qwen-max",
    "qwen/qwen3-235b-a22b-2507",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-chat",
    "rh-llm-o/rh-t-55",
    "rh-llm-o/rh-t-54",
    "rh-llm-g/rh-g-flash-preview-3",
    "rh-llm-g/rh-g-pro-preview-31",
]

AUTO_STYLE = "不指定（根据标题自动设计）"
LEGACY_AUTO_STYLES = {"不指定（根据主题自动设计）"}

STYLES = [
    AUTO_STYLE,
    "小红书干净高级风",
    "强冲击爆款标题风",
    "科技感教程封面风",
    "商业海报风",
    "杂志大片风",
    "可爱手账风",
    "电商产品种草风",
    "真实生活方式风",
    "自定义",
]

STYLE_PROFILES = {
    "小红书干净高级风": "自然、清爽、有生活方式审美和编辑感",
    "强冲击爆款标题风": "第一眼有吸引力，主体与标题关系有力量，整体像成熟内容海报",
    "科技感教程封面风": "专业、清晰、有科技与教程气质，视觉信息有秩序",
    "商业海报风": "完整的品牌主视觉和广告级画面，成熟、有传播感",
    "杂志大片风": "强调摄影、人物气场和编辑式排版，像真实杂志封面",
    "可爱手账风": "轻松、有亲和力、带手作或拼贴趣味，但保持完整设计感",
    "电商产品种草风": "突出产品价值、使用场景和购买吸引力，产品准确可信",
    "真实生活方式风": "自然、真实、有生活气息，像被捕捉到的优质内容瞬间",
}

SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.txt")


def load_system_prompt():
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def text_fingerprint(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:12]


def fetch_llm_models(force=False):
    now = time.time()
    cached = MODEL_CACHE.get("models")
    if not force and cached and now < float(MODEL_CACHE.get("expires_at", 0)):
        return list(cached)

    try:
        if requests is not None:
            response = requests.get(LLM_MODELS_URL, timeout=5)
            response.raise_for_status()
            data = response.json()
        else:
            request = urllib.request.Request(LLM_MODELS_URL, method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                body = response.read().decode("utf-8", errors="replace")
            data = json.loads(body)

        models = [
            str(item.get("id")).strip()
            for item in data.get("data", [])
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        ]
        if models:
            MODEL_CACHE["models"] = models
            MODEL_CACHE["expires_at"] = now + MODEL_CACHE_TTL_SECONDS
            return models
    except Exception as exc:
        print(f"[ViralCoverLLMPrompt] Failed to fetch RunningHub model list, using fallback: {type(exc).__name__}")

    return list(dict.fromkeys(FALLBACK_MODELS))


def default_model(models):
    return DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]


def normalize_chat_url(api_baseurl=None):
    api_baseurl = (api_baseurl or DEFAULT_API_BASEURL).strip().rstrip("/")
    if not api_baseurl:
        api_baseurl = DEFAULT_API_BASEURL
    if api_baseurl.endswith("/chat/completions"):
        return api_baseurl
    return f"{api_baseurl}/chat/completions"


def get_api_key(api_key=None):
    api_key = (api_key or "").strip()
    if api_key:
        return api_key

    env_key = (os.environ.get("RH_API_KEY") or os.environ.get("RUNNINGHUB_API_KEY") or "").strip()
    if env_key:
        return env_key

    try:
        from server import PromptServer

        shared_key = getattr(PromptServer.instance, "shared_api_key", None)
        if isinstance(shared_key, str) and shared_key.strip() and shared_key != "unknown":
            return shared_key.strip()
    except Exception:
        pass

    return ""


def tensor_to_jpeg_data_url(image):
    data_url, _ = tensor_to_jpeg_data_url_with_size(image)
    return data_url


def tensor_to_jpeg_data_url_with_size(image):
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
    return f"data:image/jpeg;base64,{encoded}", pil_image.size


def remove_think_tags(text):
    if not isinstance(text, str):
        return text

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|begin_of_box\|>|<\|end_of_box\|>", "", text)
    return text.strip()


def sanitize_for_image_generation(text):
    return text.strip() if isinstance(text, str) else text


def build_user_prompt(封面风格, 主题关键词, 封面标题, 自定义要求, has_image=False):
    image_notice = (
        "已上传参考图片，请先理解图片，再让图片服务于封面标题。"
        if has_image
        else "没有参考图片，请围绕封面标题自主创造合适的主体、场景和设计形式。"
    )
    封面风格 = (封面风格 or AUTO_STYLE).strip()
    自定义要求 = (自定义要求 or "").strip()
    主题关键词 = (主题关键词 or "").strip()
    封面标题 = (封面标题 or "").strip()

    if 封面风格 == AUTO_STYLE or 封面风格 in LEGACY_AUTO_STYLES:
        style_profile = "不预设风格，请根据标题、可选主题和参考图自行选择最合适的完整封面设计"
    elif 封面风格 == "自定义":
        style_profile = (
            "按补充要求中的自定义审美方向执行，但不要套用固定模板"
            if 自定义要求
            else "不预设风格，请根据标题和参考图自行设计"
        )
    else:
        profile = STYLE_PROFILES.get(封面风格, 封面风格)
        style_profile = f"可参考{封面风格}的审美气质（{profile}），但不要套用固定模板"

    return f"""请像商业封面视觉导演一样理解本次需求，并输出一段完整、具体、可执行的中文生图指令。
封面标题：{封面标题}
可选主题信息：{主题关键词 or "未提供，请从标题和参考图判断"}
审美方向：{style_profile}
补充要求：{自定义要求 or "无"}
{image_notice}
不要复述这些字段，不要写分析过程；要给出完整封面方案，包括主视觉、标题版式、辅助信息区、场景光影和成品质感。"""


def post_chat_completion(chat_url, headers, payload, timeout=180):
    last_error = None
    for attempt in range(CHAT_MAX_RETRIES):
        if attempt > 0:
            wait = min(2 ** attempt, 5)
            print(f"[ViralCoverLLMPrompt] Chat retry {attempt + 1}/{CHAT_MAX_RETRIES} in {wait}s...")
            time.sleep(wait)

        if requests is not None:
            response = requests.post(chat_url, headers=headers, json=payload, timeout=timeout)
            try:
                data = response.json()
            except ValueError as exc:
                raise RuntimeError(f"RunningHub LLM 返回了非 JSON 内容：{response.text[:200]}") from exc

            status_code = response.status_code
            response_text = response.text
        else:
            data_bytes = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(chat_url, data=data_bytes, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    status_code = response.status
                    response_text = response.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as exc:
                status_code = exc.code
                response_text = exc.read().decode("utf-8", errors="replace")

            try:
                data = json.loads(response_text)
            except ValueError as exc:
                raise RuntimeError(f"RunningHub LLM 返回了非 JSON 内容：{response_text[:200]}") from exc

        if status_code == 200:
            return data

        message = data.get("error") or data.get("message") or data.get("msg") or response_text[:200]
        if status_code == 401 and "auth_apikey_missing" in str(message):
            raise RuntimeError(
                "当前环境没有提供 RunningHub API Key。RunningHub 托管环境通常会自动注入；"
                "普通本地 ComfyUI 请在 api_key 中填写 Key，或设置 RH_API_KEY 环境变量。"
            )
        last_error = RuntimeError(f"RunningHub LLM 调用失败：HTTP {status_code}: {message}")
        if status_code >= 500 or status_code == 429:
            continue
        raise last_error

    raise last_error or RuntimeError("RunningHub LLM 调用失败。")


class ViralCoverLLMPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        models = fetch_llm_models()
        return {
            "required": {
                "model": (models, {"default": default_model(models)}),
                "封面标题": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "加载图像": ("IMAGE",),
                "封面风格": (STYLES, {"default": AUTO_STYLE}),
                "主题关键词": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "api_baseurl": ("STRING", {"default": DEFAULT_API_BASEURL, "multiline": False}),
                "自定义要求": ("STRING", {"default": "", "multiline": True}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("封面提示词",)
    FUNCTION = "generate"
    CATEGORY = "爆款封面生成"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Use a JSON-safe token because some hosted ComfyUI cache layers normalize NaN.
        return time.time_ns()

    def generate(
        self,
        model,
        封面标题,
        加载图像=None,
        封面风格=AUTO_STYLE,
        主题关键词="",
        api_key="",
        api_baseurl=DEFAULT_API_BASEURL,
        自定义要求="",
        temperature=0.4,
        max_tokens=2048,
    ):
        has_image = 加载图像 is not None
        if not str(封面标题 or "").strip():
            raise RuntimeError("请填写封面标题。")

        role = load_system_prompt()
        prompt = build_user_prompt(封面风格, 主题关键词, 封面标题, 自定义要求, has_image=has_image)
        headers = {
            "Content-Type": "application/json",
        }
        api_key_value = get_api_key(api_key)
        if api_key_value:
            headers["Authorization"] = f"Bearer {api_key_value}"

        user_content = prompt
        image_size = None
        if has_image:
            image_url, image_size = tensor_to_jpeg_data_url_with_size(加载图像)
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
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
        }

        chat_url = normalize_chat_url(api_baseurl)
        print(
            json.dumps(
                {
                    "node": "ViralCoverLLMPrompt",
                    "model": model,
                    "cover_style": 封面风格,
                    "chat_url": chat_url,
                    "node_file": str(Path(__file__).resolve()),
                    "system_prompt_path": str(SYSTEM_PROMPT_PATH.resolve()),
                    "system_prompt_hash": text_fingerprint(role),
                    "has_image": has_image,
                    "image_size": image_size,
                    "content_type": "multimodal" if has_image else "text",
                },
                ensure_ascii=False,
            )
        )

        try:
            data = post_chat_completion(chat_url, headers, payload)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"RunningHub LLM 网络请求失败：{exc}") from exc

        choices = data.get("choices")
        if not choices:
            raise RuntimeError("RunningHub LLM 没有返回可用内容。")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("RunningHub LLM 返回格式不正确。")

        message = first_choice.get("message", {})
        content = message.get("content") if isinstance(message, dict) else None
        content = content or first_choice.get("text")
        if isinstance(content, list):
            content = "\n".join(
                item.get("text", "") for item in content if isinstance(item, dict) and item.get("text")
            )

        if not content:
            raise RuntimeError("RunningHub LLM 返回内容为空。")

        result = sanitize_for_image_generation(remove_think_tags(str(content)))
        print(f"[ViralCoverLLMPrompt] RH LLM request finished at {int(time.time())}")
        print(
            json.dumps(
                {
                    "model": model,
                    "cover_style": 封面风格,
                    "output_length": len(result),
                    "output_hash": text_fingerprint(result),
                    "has_image": has_image,
                },
                ensure_ascii=False,
            )
        )
        return (result,)


NODE_CLASS_MAPPINGS = {
    "ViralCoverLLMPrompt": ViralCoverLLMPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ViralCoverLLMPrompt": "YM-爆款封面",
}
