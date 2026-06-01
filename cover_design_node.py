from __future__ import annotations

from typing import Tuple


class RHCoverDesignNode:
    """
    封面图设计节点（示例壳节点）。

    左侧输入 image，右侧输出 image。
    当前节点不执行实际图像生成，仅完成参数收集与透传，
    便于在无本地 ComfyUI 环境时先完成节点结构开发。
    """

    CATEGORY = "RunningHub/Cover"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "design_cover"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "platform": (
                    ["抖音", "快手", "小红书", "视频号"],
                    {"default": "抖音"},
                ),
                "cover_ratio": (
                    ["1:1", "3:4", "4:3", "9:16", "16:9"],
                    {"default": "1:1"},
                ),
                "llm_model": (
                    [
                        "gpt-4o-mini",
                        "gpt-4.1-mini",
                        "qwen-plus",
                        "deepseek-chat",
                    ],
                    {"default": "gpt-4o-mini"},
                ),
                "design_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "请生成适合所选平台的吸睛封面风格说明。",
                    },
                ),
            }
        }

    def design_cover(
        self,
        image,
        platform: str,
        cover_ratio: str,
        llm_model: str,
        design_prompt: str,
    ) -> Tuple:
        # 预留：后续可在此拼接 payload 并调用 RunningHub OpenAPI。
        _ = (platform, cover_ratio, llm_model, design_prompt)
        return (image,)


NODE_CLASS_MAPPINGS = {
    "RHCoverDesignNode": RHCoverDesignNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RHCoverDesignNode": "RH 封面图设计",
}
