from viral_cover_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, ViralCoverLLMPrompt


def main():
    node = ViralCoverLLMPrompt()
    input_types = node.INPUT_TYPES()
    print("节点注册名:")
    print(NODE_DISPLAY_NAME_MAPPINGS["ViralCoverLLMPrompt"])
    print()
    print("节点数量:")
    print(len(NODE_CLASS_MAPPINGS))
    print()
    print("必填输入:")
    print(", ".join(input_types["required"].keys()))
    print()
    print("输出:")
    print(", ".join(node.RETURN_NAMES))


if __name__ == "__main__":
    main()
