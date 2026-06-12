import base64
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
DEFAULT_MODEL = "google/gemini-3.1-flash-lite-preview"
MODEL_CACHE = {"expires_at": 0.0, "models": None}

FALLBACK_MODELS = [
    DEFAULT_MODEL,
    "google/gemini-3.5-flash",
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

STYLES = [
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
    "小红书干净高级风": "明亮自然、干净留白、精致生活方式感、配色克制，标题清爽醒目，避免强压迫式大字和杂乱高对比元素。",
    "强冲击爆款标题风": "大主体、大标题、强对比、强情绪、强记忆点，使用聚光、反差色块、近景和醒目字效，画面一眼抓住重点；不要沿用小红书式轻盈留白和柔和表达。",
    "科技感教程封面风": "使用简洁发光界面、提示词卡片、数字光效、玻璃拟态和少量流程元素，清晰专业，避免复杂软件截图、满屏代码和元素堆叠。",
    "商业海报风": "广告级构图与专业灯光，主体明确，背景干净，标题和主体形成成熟稳定的品牌宣传版式。",
    "杂志大片风": "高级编辑感、大片光影、克制留白、统一配色和杂志式标题排版，主体有气场且画面有呼吸感。",
    "可爱手账风": "柔和配色、贴纸、便签、圆角卡片、纸张纹理和手写标签感，轻松亲切但主次清晰，避免幼稚杂乱。",
    "电商产品种草风": "产品是第一视觉中心，清晰大且材质真实，突出核心卖点和使用场景，具有精致广告感与购买吸引力。",
    "真实生活方式风": "自然真实的生活场景与瞬间感，光线可信、人物状态放松，减少摆拍和过度商业修饰，同时保持封面重点与可读性。",
}

SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.txt")


def load_system_prompt():
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


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

    return list(FALLBACK_MODELS)


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


SAFETY_REPLACEMENTS = {
    "比基尼": "日常服装",
    "泳装": "夏季服装",
    "性感": "时尚",
    "诱惑": "吸引力",
    "撩人": "有表现力",
    "暴露": "清爽",
    "裸露": "清爽",
    "透明": "轻薄质感",
    "透视": "轻薄质感",
    "深V": "简洁领口",
    "低胸": "简洁领口",
    "胸部": "上半身",
    "大胸": "身形比例自然",
    "臀部": "整体姿态",
    "美腿": "腿部线条自然",
    "腿部特写": "全身构图",
    "皮肤裸露": "服装完整",
    "身体曲线": "整体轮廓",
    "身材曲线": "整体轮廓",
    "挑逗": "自信",
}


def sanitize_for_image_generation(text):
    if not isinstance(text, str):
        return text

    sanitized = text
    for source, target in SAFETY_REPLACEMENTS.items():
        sanitized = sanitized.replace(source, target)

    safety_note = (
        "人物呈现自然得体，服装完整，避免低俗、擦边、过度暴露、身体局部特写、"
        "未成年人不适宜内容、文字乱码、水印和杂乱背景。"
    )
    if "避免低俗" not in sanitized and "服装完整" not in sanitized:
        sanitized = f"{sanitized}，{safety_note}"
    return sanitized.strip()


def ensure_active_style(text, cover_style):
    """Keep the selected style explicit even when the model omits its label."""
    if not isinstance(text, str) or not cover_style or cover_style in text:
        return text
    return f"封面风格为“{cover_style}”。{text}"


def build_user_prompt(封面风格, 主题关键词, 封面标题, 自定义要求, has_image=False):
    image_notice = (
        "本次用户已经上传了一张参考图片。你必须先识别图片里的真实主体、服装、场景、背景、构图和光线，再生成封面提示词。"
        "最终提示词必须和原图内容一致，不要编造原图里不存在的海边、沙滩、泳装、比基尼、户外阳光等元素。"
        if has_image
        else "本次用户没有上传参考图片，请直接根据文字变量生成适合文生图的封面提示词。"
    )
    if 封面风格 == "自定义":
        style_profile = 自定义要求.strip() or "高点击率商业封面风，画面清晰、有冲击力、干净高级"
    else:
        style_profile = STYLE_PROFILES.get(封面风格, 封面风格)

    style_notice = (
        f"本次请求当前唯一生效的封面风格是“{封面风格}”。具体执行规范：{style_profile}"
        "必须从头按照本次风格重新组织构图、色彩、光影、背景和标题字效，不得复用或沿用上一次运行的风格描述。"
        "最终提示词中必须明确写出本次选择的封面风格及其核心视觉特征。"
    )
    return f"""请根据我提供的图片或文字需求，结合以下变量，输出一段可直接用于 AI 绘图的完整封面提示词。
封面风格：{封面风格}
主题关键词：{主题关键词}
封面标题：{封面标题.strip() or "未提供，请根据主题自动生成一个适合封面展示的中文短标题"}
自定义要求：{自定义要求.strip() or "无"}

{image_notice}

{style_notice}

内容安全要求：如果图片里有人物，必须使用自然、得体、完整服装的中性描述，不要输出性感、诱惑、暴露、裸露、透视、低胸、比基尼、泳装、身体局部特写等容易触发生图安全审查的词。不要把普通穿搭图改写成擦边、泳装、亲密或成人化场景。

如果我提供了图片，请重点分析原图里的主体、构图、色彩、光影、背景、人物/产品状态、可复用的视觉卖点，并根据主题关键词、封面标题和封面风格生成提示词。不要因为缺少平台类型或内容类型就自行套用固定分类。
如果没有图片，请直接根据主题关键词、封面标题、封面风格和自定义要求生成适合文生图的封面提示词。输出时只给最终提示词，不要解释过程。"""


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
                "封面风格": (STYLES,),
                "主题关键词": ("STRING", {"default": "7天学会AI绘画", "multiline": False}),
                "封面标题": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "加载图像": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "api_baseurl": ("STRING", {"default": DEFAULT_API_BASEURL, "multiline": False}),
                "自定义要求": ("STRING", {"default": "", "multiline": True}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("系统提示词",)
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
        封面风格,
        主题关键词,
        封面标题,
        加载图像=None,
        api_key="",
        api_baseurl=DEFAULT_API_BASEURL,
        自定义要求="",
        temperature=0.4,
        max_tokens=2048,
    ):
        has_image = 加载图像 is not None
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
                    "has_image": has_image,
                    "image_size": image_size,
                    "content_type": "multimodal" if has_image else "text",
                },
                ensure_ascii=False,
            )
        )

        try:
            data = post_chat_completion(chat_url, headers, payload)
        except Exception as exc:
            raise RuntimeError(f"RunningHub LLM 网络请求失败：{exc}") from exc

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

        result = sanitize_for_image_generation(remove_think_tags(str(content)))
        result = ensure_active_style(result, 封面风格)
        print(f"[ViralCoverLLMPrompt] RH LLM request finished at {int(time.time())}")
        print(json.dumps({"model": model, "output_length": len(result), "has_image": has_image}, ensure_ascii=False))
        return (result,)


NODE_CLASS_MAPPINGS = {
    "ViralCoverLLMPrompt": ViralCoverLLMPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ViralCoverLLMPrompt": "YM-爆款封面",
}
