import json
import faiss
import pickle
import numpy as np
from FlagEmbedding import FlagModel, FlagReranker
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("加载测试集...")
with open("/root/autodl-tmp/data/test_set.json", "r") as f:
    test_data = json.load(f)

print(f"测试集: {len(test_data)} 条")

print("加载知识库...")
index = faiss.read_index("/root/autodl-tmp/data/law_faiss.index")
with open("/root/autodl-tmp/data/law_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("加载 embedding 模型...")
embed_model = FlagModel(
    "/root/autodl-tmp/models/bge-large-zh-v1.5",
    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关法律条文："
)

print("加载 reranker...")
reranker = FlagReranker("/root/autodl-tmp/models/bge-reranker-v2-m3", use_fp16=True)

def retrieve(query, top_k_initial=20, top_k_final=5, use_rerank=True):
    q_emb = embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k_initial)
    
    candidates = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            candidates.append(chunks[idx])
    
    if not use_rerank or len(candidates) == 0:
        return candidates[:top_k_final]
    
    pairs = [[query, doc] for doc in candidates]
    rerank_scores = reranker.compute_score(pairs)
    if isinstance(rerank_scores, float):
        rerank_scores = [rerank_scores]
    
    ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k_final]]

def load_model(model_path):
    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer

def generate_answer(model, tokenizer, question, contexts=None, max_new_tokens=512):
    if contexts:
        context_str = "\n\n".join(contexts)
        prompt = f"请根据以下法律参考资料回答问题。如果参考资料不足以回答，请说明。\n\n【参考资料】\n{context_str}\n\n【问题】\n{question}\n\n【回答】"
    else:
        prompt = f"请回答以下法律问题：\n\n{question}"
    
    messages = [
        {"role": "system", "content": "你是一个专业的中国法律助手。请基于法律法规准确回答问题。/no_think"},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if "<think>" in response:
        response = response.split("</think>")[-1].strip()
    return response

results = {
    "config1_base": [],
    "config2_sft": [],
    "config3_rag": [],
    "config4_sft_rag": [],
}

questions = [item["input"] for item in test_data]
references = [item["output"] for item in test_data]

# --- Config 1: 纯基座模型 ---
print("\n===== Config 1: 纯基座模型 =====")
model, tokenizer = load_model("/root/autodl-tmp/models/Qwen3-8B")

for i, q in enumerate(questions):
    answer = generate_answer(model, tokenizer, q)
    results["config1_base"].append({
        "question": q, "answer": answer, "reference": references[i], "contexts": []
    })
    if (i+1) % 20 == 0:
        print(f"  Config1 进度: {i+1}/{len(questions)}")

del model, tokenizer
torch.cuda.empty_cache()

# --- Config 2: 纯 LoRA 微调 ---
print("\n===== Config 2: 纯 LoRA 微调 =====")
model, tokenizer = load_model("/root/autodl-tmp/models/Qwen3-8B-Law-Merged")

for i, q in enumerate(questions):
    answer = generate_answer(model, tokenizer, q)
    results["config2_sft"].append({
        "question": q, "answer": answer, "reference": references[i], "contexts": []
    })
    if (i+1) % 20 == 0:
        print(f"  Config2 进度: {i+1}/{len(questions)}")

del model, tokenizer
torch.cuda.empty_cache()

# --- Config 3: 纯 RAG ---
print("\n===== Config 3: 纯 RAG =====")
model, tokenizer = load_model("/root/autodl-tmp/models/Qwen3-8B")

for i, q in enumerate(questions):
    contexts = retrieve(q, use_rerank=False)
    answer = generate_answer(model, tokenizer, q, contexts=contexts)
    results["config3_rag"].append({
        "question": q, "answer": answer, "reference": references[i], "contexts": contexts
    })
    if (i+1) % 20 == 0:
        print(f"  Config3 进度: {i+1}/{len(questions)}")

del model, tokenizer
torch.cuda.empty_cache()

# --- Config 4: LoRA + RAG + Reranker ---
print("\n===== Config 4: LoRA + RAG + Reranker =====")
model, tokenizer = load_model("/root/autodl-tmp/models/Qwen3-8B-Law-Merged")

for i, q in enumerate(questions):
    contexts = retrieve(q, use_rerank=True)
    answer = generate_answer(model, tokenizer, q, contexts=contexts)
    results["config4_sft_rag"].append({
        "question": q, "answer": answer, "reference": references[i], "contexts": contexts
    })
    if (i+1) % 20 == 0:
        print(f"  Config4 进度: {i+1}/{len(questions)}")

del model, tokenizer
torch.cuda.empty_cache()

with open("/root/autodl-tmp/data/inference_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n所有推理完成！")