"""
Maya1 模型下载脚本
用法: python download_maya1.py [--model-dir PATH] [--force]
"""
import os
import sys
import argparse


def get_default_model_dir():
    """获取默认模型目录"""
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "runtime", "python", "models", "maya1")


def is_model_ready(model_dir):
    """检查模型文件是否已下载完成"""
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]
    for f in required_files:
        if not os.path.isfile(os.path.join(model_dir, f)):
            return False

    # 检查至少有一个 safetensors 权重分片
    has_weights = any(
        name.endswith(".safetensors")
        for name in os.listdir(model_dir)
        if name.startswith("model-")
    )
    return has_weights


def download_model(model_dir, repo_id="maya-research/maya1"):
    """从 HuggingFace 下载 Maya1 模型"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("错误: 需要安装 huggingface_hub")
        print("  pip install huggingface_hub")
        sys.exit(1)

    print(f"模型仓库: {repo_id}")
    print(f"下载目录: {model_dir}")
    print("开始下载（模型约 6GB，请耐心等待）...")
    print()

    snapshot_download(
        repo_id=repo_id,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
    )

    print()
    print("模型下载完成！")


def main():
    parser = argparse.ArgumentParser(description="下载 Maya1 模型")
    parser.add_argument(
        "--model-dir",
        default=get_default_model_dir(),
        help="模型保存路径",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载（即使模型已存在）",
    )
    args = parser.parse_args()

    if not args.force and is_model_ready(args.model_dir):
        print(f"模型已存在: {args.model_dir}")
        print("  使用 --force 强制重新下载")
        return

    os.makedirs(args.model_dir, exist_ok=True)
    download_model(args.model_dir)


if __name__ == "__main__":
    main()
