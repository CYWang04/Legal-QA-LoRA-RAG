import os

# ============================================================
# 使用前请根据你的环境修改以下路径
# ============================================================

# 基座模型路径（从 ModelScope 或 HuggingFace 下载）
BASE_MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"

# LoRA 微调后合并的模型路径（训练完成后生成）
MERGED_MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B-Law-Merged"

# BGE Embedding 模型路径
EMBEDDING_MODEL_PATH = "/root/autodl-tmp/models/bge-large-zh-v1.5"

# BGE Reranker 模型路径
RERANKER_MODEL_PATH = "/root/autodl-tmp/models/bge-reranker-v2-m3"

# LoRA adapter 保存路径（训练过程中生成）
LORA_SAVE_PATH = "/root/autodl-tmp/saves/qwen3-8b-law-lora"

# 数据目录（存放原始数据集、处理后的数据、FAISS 索引等）
DATA_DIR = "/root/autodl-tmp/data"

# LLaMA-Factory 目录
LLAMAFACTORY_DIR = "/root/autodl-tmp/LLaMA-Factory"