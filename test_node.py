import tempfile
import time
import unittest
import base64
from io import BytesIO
from pathlib import Path

import viral_cover_nodes as module


class ViralCoverNodeTests(unittest.TestCase):
    def test_current_system_prompt_is_the_rewritten_version(self):
        prompt = module.load_system_prompt()
        self.assertTrue(prompt.startswith("你是一名商业封面视觉导演"))
        self.assertIn("封面标题是本次设计的核心", prompt)
        self.assertIn("也是唯一必定存在的创作信息", prompt)
        self.assertIn("只有标题、没有图片和其他文案时", prompt)
        self.assertIn("必须足够具体，让普通生图模型也能做出封面", prompt)
        self.assertIn("一个强主视觉，一个醒目的标题视觉", prompt)
        self.assertIn("人物不能只是远远站着或坐着", prompt)
        self.assertIn("通常控制在 300 到 520 个汉字", prompt)
        self.assertNotIn("负面限制必须包含", prompt)
        self.assertNotIn("封面必须有主体、有内容、有情绪、有动作、有道具", prompt)

    def test_user_prompt_requires_a_complete_cover_brief(self):
        strong = module.build_user_prompt("强冲击爆款标题风", "通用主题", "测试标题", "", True)
        cute = module.build_user_prompt("可爱手账风", "通用主题", "测试标题", "", True)
        self.assertLess(len(strong), 360)
        self.assertLess(len(cute), 360)
        self.assertIn("可选主题信息：通用主题", strong)
        self.assertIn("封面标题：测试标题", strong)
        self.assertIn("已上传参考图片", strong)
        self.assertIn("完整封面方案", strong)
        self.assertIn("辅助信息区", strong)
        self.assertNotIn("红白粗描边巨字", strong)
        self.assertNotIn("格纹相框", cute)
        self.assertNotIn("必须从头按照", strong)

    def test_auto_style_is_default_and_does_not_impose_a_preset(self):
        self.assertEqual(module.STYLES[0], module.AUTO_STYLE)
        prompt = module.build_user_prompt(module.AUTO_STYLE, "旅行记录", "", "", False)
        self.assertIn("不预设风格", prompt)
        self.assertNotIn("小红书", prompt)
        self.assertNotIn("强冲击", prompt)

        legacy_prompt = module.build_user_prompt(
            "不指定（根据主题自动设计）", "旅行记录", "旅行的一天", "", False
        )
        self.assertIn("不预设风格", legacy_prompt)

    def test_custom_style_requirement_is_not_duplicated(self):
        prompt = module.build_user_prompt("自定义", "", "旅行的一天", "电影海报质感", False)
        self.assertEqual(prompt.count("电影海报质感"), 1)

    def test_image_encoder_creates_a_valid_jpeg_data_url(self):
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            self.skipTest(str(exc))

        image = np.zeros((1, 8, 12, 3), dtype=np.float32)
        image[:, :, :, 0] = 1.0
        data_url, size = module.tensor_to_jpeg_data_url_with_size(image)
        self.assertEqual(size, (12, 8))
        self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))
        encoded = data_url.split(",", 1)[1]
        decoded = Image.open(BytesIO(base64.b64decode(encoded)))
        self.assertEqual(decoded.size, (12, 8))
        self.assertEqual(decoded.mode, "RGB")

    def test_model_and_cover_title_are_required_in_the_node_ui(self):
        original_fetch = module.fetch_llm_models
        try:
            module.fetch_llm_models = lambda: [module.DEFAULT_MODEL]
            input_types = module.ViralCoverLLMPrompt.INPUT_TYPES()
        finally:
            module.fetch_llm_models = original_fetch

        self.assertEqual(set(input_types["required"]), {"model", "封面标题"})
        for name in ("加载图像", "封面风格", "主题关键词", "自定义要求"):
            self.assertIn(name, input_types["optional"])

    def test_cover_title_is_required_and_other_content_is_optional(self):
        node = module.ViralCoverLLMPrompt()
        with self.assertRaisesRegex(RuntimeError, "请填写封面标题"):
            node.generate("test-model", "")

        original_post = module.post_chat_completion
        try:
            module.post_chat_completion = lambda *_args, **_kwargs: {
                "choices": [{"message": {"content": "围绕指定标题完成一张有设计感的封面。"}}]
            }
            result = node.generate("test-model", "旅行的一天")[0]
        finally:
            module.post_chat_completion = original_post

        self.assertIn("指定标题", result)

    def test_output_is_not_prefixed_with_style_label(self):
        cleaned = module.sanitize_for_image_generation("采用克制的编辑式构图")
        self.assertEqual(cleaned, "采用克制的编辑式构图")
        self.assertFalse(hasattr(module, "ensure_active_style"))

    def test_safety_cleanup_does_not_append_negative_prompt(self):
        cleaned = module.sanitize_for_image_generation("人物在室内展示教程卡片")
        self.assertEqual(cleaned, "人物在室内展示教程卡片")
        self.assertNotIn("避免低俗", cleaned)

    def test_registration(self):
        self.assertEqual(module.NODE_DISPLAY_NAME_MAPPINGS["ViralCoverLLMPrompt"], "YM-爆款封面")
        self.assertTrue(module.ViralCoverLLMPrompt.OUTPUT_NODE)
        self.assertEqual(module.ViralCoverLLMPrompt.RETURN_TYPES, ("STRING",))
        self.assertEqual(module.ViralCoverLLMPrompt.RETURN_NAMES, ("封面提示词",))

    def test_default_model_prioritizes_gemini_vision_model(self):
        self.assertEqual(module.DEFAULT_MODEL, "google/gemini-3.5-flash")
        self.assertEqual(module.default_model([module.DEFAULT_MODEL, "other-model"]), module.DEFAULT_MODEL)
        self.assertEqual(len(module.FALLBACK_MODELS), len(set(module.FALLBACK_MODELS)))

    def test_api_url_and_explicit_key(self):
        self.assertEqual(
            module.normalize_chat_url("https://example.com/v1"),
            "https://example.com/v1/chat/completions",
        )
        self.assertEqual(
            module.normalize_chat_url("https://example.com/v1/chat/completions"),
            "https://example.com/v1/chat/completions",
        )
        self.assertEqual(module.get_api_key("  test-key  "), "test-key")

    def test_missing_api_key_error_is_clear(self):
        class FakeResponse:
            status_code = 401
            text = '{"error":{"message":"api key is required","code":"auth_apikey_missing"}}'

            @staticmethod
            def json():
                return {
                    "error": {
                        "message": "api key is required",
                        "code": "auth_apikey_missing",
                    }
                }

        class FakeRequests:
            @staticmethod
            def post(*_args, **_kwargs):
                return FakeResponse()

        original_requests = module.requests
        try:
            module.requests = FakeRequests()
            with self.assertRaisesRegex(RuntimeError, "普通本地 ComfyUI 请在 api_key 中填写 Key"):
                module.post_chat_completion("https://example.com/v1/chat/completions", {}, {})
        finally:
            module.requests = original_requests

    def test_each_queue_gets_a_new_cache_token(self):
        first = module.ViralCoverLLMPrompt.IS_CHANGED()
        time.sleep(0.001)
        second = module.ViralCoverLLMPrompt.IS_CHANGED()
        self.assertNotEqual(first, second)

    def test_style_image_and_system_prompt_refresh_together(self):
        calls = []
        displayed = []

        def fake_post_chat_completion(_chat_url, _headers, payload, timeout=180):
            del timeout
            calls.append(payload)
            system_prompt = payload["messages"][0]["content"]
            user_content = payload["messages"][1]["content"]
            prompt = user_content[0]["text"]
            if "第一眼有吸引力" in prompt:
                answer = f"{system_prompt}：大主体、大标题、强对比的封面方案。"
            else:
                answer = f"{system_prompt}：明亮留白、清爽高级的小红书封面方案。"
            return {"choices": [{"message": {"content": answer}}]}

        def show_text(value):
            displayed.append(value)

        original_system_prompt_path = module.SYSTEM_PROMPT_PATH
        original_post = module.post_chat_completion
        original_image_encoder = module.tensor_to_jpeg_data_url_with_size
        image = object()

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                prompt_path = Path(temp_dir) / "system_prompt.txt"
                module.SYSTEM_PROMPT_PATH = prompt_path
                module.post_chat_completion = fake_post_chat_completion
                module.tensor_to_jpeg_data_url_with_size = lambda _image: (
                    "data:image/jpeg;base64,TEST_IMAGE",
                    (12, 12),
                )
                node = module.ViralCoverLLMPrompt()

                prompt_path.write_text("系统提示词版本一", encoding="utf-8")
                first = node.generate(
                    "test-model",
                    "同一个标题",
                    加载图像=image,
                    封面风格="小红书干净高级风",
                    主题关键词="同一个主题",
                    api_baseurl="https://example.com/v1",
                    max_tokens=512,
                )[0]
                show_text(first)

                second = node.generate(
                    "test-model",
                    "同一个标题",
                    加载图像=image,
                    封面风格="强冲击爆款标题风",
                    主题关键词="同一个主题",
                    api_baseurl="https://example.com/v1",
                    max_tokens=512,
                )[0]
                show_text(second)

                prompt_path.write_text("系统提示词版本二", encoding="utf-8")
                third = node.generate(
                    "test-model",
                    "同一个标题",
                    加载图像=image,
                    封面风格="强冲击爆款标题风",
                    主题关键词="同一个主题",
                    api_baseurl="https://example.com/v1",
                    max_tokens=512,
                )[0]
                show_text(third)
        finally:
            module.SYSTEM_PROMPT_PATH = original_system_prompt_path
            module.post_chat_completion = original_post
            module.tensor_to_jpeg_data_url_with_size = original_image_encoder

        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]["messages"][0]["content"], "系统提示词版本一")
        self.assertEqual(calls[1]["messages"][0]["content"], "系统提示词版本一")
        self.assertEqual(calls[2]["messages"][0]["content"], "系统提示词版本二")

        first_content = calls[0]["messages"][1]["content"]
        second_content = calls[1]["messages"][1]["content"]
        third_content = calls[2]["messages"][1]["content"]
        self.assertIsInstance(first_content, list)
        self.assertIsInstance(second_content, list)
        self.assertIsInstance(third_content, list)
        self.assertTrue(first_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertTrue(second_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertTrue(third_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertIn("生活方式审美和编辑感", first_content[0]["text"])
        self.assertIn("第一眼有吸引力", second_content[0]["text"])
        self.assertIn("第一眼有吸引力", third_content[0]["text"])

        self.assertIn("系统提示词版本一", first)
        self.assertIn("系统提示词版本一", second)
        self.assertIn("系统提示词版本二", third)
        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)
        self.assertEqual(displayed, [first, second, third])


if __name__ == "__main__":
    unittest.main(verbosity=2)
