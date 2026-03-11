
# 基于 LoRA 与 RAG 混合架构的法律大模型问答系统

本项目是本人学习大语言模型微调与 RAG 检索增强技术的实践项目。针对通用大模型在法律场景下存在的"专业表达不足"与"法条引用幻觉"问题，设计并实现了一套融合 LoRA 参数高效微调与 RAG 检索增强的混合架构方案。

> ⚠️ **路径说明：** 由于近期忙于准备研究生复试，`scripts/` 下的 `.py` 文件和 `configs/` 下的 `.yaml` 文件中的路径仍为本人实验环境中的 `/root/autodl-tmp/...`，计划于 2026 年 4 月后统一重构为可配置路径。如需复现实验，请先将代码中的路径替换为您自己的实际路径。

## 项目架构

```
用户提问
  │
  ├──→ BGE Embedding 向量化
  │         │
  │         ▼
  │    FAISS 向量检索（粗排 Top-20）
  │         │
  │         ▼
  │    BGE-Reranker 重排序（精排 Top-5）
  │         │
  │         ▼
  │    检索到的法条文本
  │         │
  ▼         ▼
  拼接为 Prompt ──→ Qwen3-8B (LoRA 微调后) ──→ 生成回答
```

- **LoRA 微调**主要解决专业性问题：基于 DISC-Law-SFT 数据集对 Qwen3-8B 进行参数高效微调，让模型学会法律领域的回答风格与专业术语。
- **RAG 检索增强**主要解决幻觉问题：回答时先从法条知识库中检索真实法律条文作为参考依据，避免模型凭记忆编造不存在的法条。

## 实验结果

在 200 条法律问答测试集上设计了四组对照实验（基座 / 纯微调 / 纯 RAG / 混合架构），完整的推理结果保存在 `results/inference_results.json`，评测指标详见 `results/` 目录。

## 环境要求

- GPU：至少 24GB 显存
- Python >= 3.10
- PyTorch >= 2.6.0
- CUDA >= 12.4

## 安装与准备

### 1. 克隆本仓库

```bash
git clone https://github.com/CYWang04/Legal-QA-LoRA-RAG.git
cd Legal-QA-LoRA-RAG
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装 LLaMA-Factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### 4. 下载模型

从 [ModelScope](https://modelscope.cn) 下载以下三个模型：

```bash
modelscope download --model Qwen/Qwen3-8B --local_dir /你的路径/models/Qwen3-8B
modelscope download --model BAAI/bge-large-zh-v1.5 --local_dir /你的路径/models/bge-large-zh-v1.5
modelscope download --model BAAI/bge-reranker-v2-m3 --local_dir /你的路径/models/bge-reranker-v2-m3
```

### 5. 下载数据集

从 [HuggingFace](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT) 下载 DISC-Law-SFT 数据集：

```bash
export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像加速
wget https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT/resolve/main/DISC-Law-SFT-Pair-QA-released.jsonl
wget https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT/resolve/main/DISC-Law-SFT-Triplet-QA-released.jsonl
```

## 运行流程

按以下顺序执行：

```bash
# Step 1: 数据准备（生成训练集、测试集、RAG 知识库）
python scripts/prepare_data.py

# Step 2: 注册数据集到 LLaMA-Factory
python scripts/register_dataset.py

# Step 3: LoRA 微调
llamafactory-cli train configs/law_lora_sft.yaml

# Step 4: 合并 LoRA 权重到基座模型
llamafactory-cli export configs/merge_lora.yaml

# Step 5: 构建 RAG 向量知识库（FAISS 索引）
python scripts/build_rag.py

# Step 6: 四组对照实验推理
python scripts/run_inference.py

# Step 7: 计算评测指标（ROUGE-L / BLEU-4）
python scripts/evaluate_metrics.py
```

## 项目结构

```
Legal-QA-LoRA-RAG/
├── README.md
├── requirements.txt
├── configs/
│   ├── law_lora_sft.yaml          # LoRA 微调训练配置
│   └── merge_lora.yaml            # LoRA 权重合并配置
├── scripts/
│   ├── prepare_data.py            # 数据准备：切分训练集/测试集，提取知识库
│   ├── register_dataset.py        # 注册数据集到 LLaMA-Factory
│   ├── build_rag.py               # 文本分块、向量化、构建 FAISS 索引
│   ├── run_inference.py           # 四组对照实验（基座/纯微调/纯RAG/混合）
│   └── evaluate_metrics.py        # ROUGE-L 与 BLEU-4 评测
├── saves/                         # LoRA adapter 权重与训练日志
├── data/                          # 处理后的数据、FAISS 索引等
└── results/
    └── inference_results.json     # 四组实验的完整推理结果
```

## 技术栈

| 组件 | 工具 | 用途 |
|------|------|------|
| 基座模型 | [Qwen3-8B](https://modelscope.cn/models/Qwen/Qwen3-8B) | 大语言模型 |
| 微调框架 | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | LoRA 参数高效微调 |
| 微调方法 | LoRA (rank=16, alpha=32, target=all) | 冻结原模型，只训练低秩增量矩阵 |
| 向量化 | [BGE-large-zh-v1.5](https://modelscope.cn/models/BAAI/bge-large-zh-v1.5) | 将文本编码为 1024 维语义向量 |
| 向量检索 | FAISS (IndexFlatIP) | 高效向量相似度搜索（粗排） |
| 重排序 | [BGE-Reranker-v2-m3](https://modelscope.cn/models/BAAI/bge-reranker-v2-m3) | Cross-Encoder 精排 |
| 训练数据 | [DISC-Law-SFT](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT) Pair-QA | 5000 条法律问答指令数据 |
| 知识库 | DISC-Law-SFT Triplet-QA | 法条原文（reference 字段） |

## 致谢

- [DISC-LawLLM](https://github.com/FudanDISC/DISC-LawLLM) — 复旦大学 DISC 实验室，本项目使用了 DISC-Law-SFT 数据集
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — 大模型高效微调框架
- [Qwen3](https://github.com/QwenLM/Qwen3) — 阿里通义千问团队
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) — 北京智源研究院，提供 BGE 系列模型
