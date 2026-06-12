import tempfile
import time
import unittest
from pathlib import Path

import viral_cover_nodes as module


class ViralCoverNodeTests(unittest.TestCase):
    def test_current_system_prompt_is_the_rewritten_version(self):
        prompt = module.load_system_prompt()
        self.assertTrue(prompt.startswith("你是一名爆款封面提示词优化助手"))
        self.assertIn("重新设计一张完整封面", prompt)
        self.assertIn("普通封面控制在 250 到 500 字左右", prompt)
        self.assertIn("复杂图生图封面控制在 500 到 700 字左右", prompt)
        self.assertIn("不要单独输出一大段负面提示词", prompt)
        self.assertNotIn("负面限制必须包含", prompt)

    def test_safety_cleanup_does_not_append_negative_prompt(self):
        cleaned = module.sanitize_for_image_generation("人物在室内展示教程卡片")
        self.assertEqual(cleaned, "人物在室内展示教程卡片")
        self.assertNotIn("避免低俗", cleaned)

    def test_registration(self):
        self.assertEqual(module.NODE_DISPLAY_NAME_MAPPINGS["ViralCoverLLMPrompt"], "YM-爆款封面")
        self.assertTrue(module.ViralCoverLLMPrompt.OUTPUT_NODE)
        self.assertEqual(module.ViralCoverLLMPrompt.RETURN_TYPES, ("STRING",))

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
            if "强冲击爆款标题风" in prompt:
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
                    "小红书干净高级风",
                    "同一个主题",
                    "同一个标题",
                    image,
                    "",
                    "https://example.com/v1",
                    "",
                    0.4,
                    512,
                )[0]
                show_text(first)

                second = node.generate(
                    "test-model",
                    "强冲击爆款标题风",
                    "同一个主题",
                    "同一个标题",
                    image,
                    "",
                    "https://example.com/v1",
                    "",
                    0.4,
                    512,
                )[0]
                show_text(second)

                prompt_path.write_text("系统提示词版本二", encoding="utf-8")
                third = node.generate(
                    "test-model",
                    "强冲击爆款标题风",
                    "同一个主题",
                    "同一个标题",
                    image,
                    "",
                    "https://example.com/v1",
                    "",
                    0.4,
                    512,
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
        self.assertIn("小红书干净高级风", first_content[0]["text"])
        self.assertIn("强冲击爆款标题风", second_content[0]["text"])
        self.assertIn("强冲击爆款标题风", third_content[0]["text"])

        self.assertIn("小红书干净高级风", first)
        self.assertIn("强冲击爆款标题风", second)
        self.assertIn("强冲击爆款标题风", third)
        self.assertIn("系统提示词版本一", first)
        self.assertIn("系统提示词版本一", second)
        self.assertIn("系统提示词版本二", third)
        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)
        self.assertEqual(displayed, [first, second, third])


if __name__ == "__main__":
    unittest.main(verbosity=2)
