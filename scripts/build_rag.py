import json
import faiss
import numpy as np
from FlagEmbedding import FlagModel
import pickle

print("加载 embedding 模型...")
embed_model = FlagModel(
    "/root/autodl-tmp/models/bge-large-zh-v1.5",
    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关法律条文："
)

print("加载知识库...")
docs = []
with open("/root/autodl-tmp/data/law_knowledge_base.jsonl", "r") as f:
    for line in f:
        item = json.loads(line.strip())
        docs.append(item["text"])

print(f"知识库文档数: {len(docs)}")

chunks = []
chunk_size = 512
overlap = 100

for doc in docs:
    if len(doc) <= chunk_size:
        chunks.append(doc)
    else:
        for start in range(0, len(doc), chunk_size - overlap):
            chunk = doc[start:start + chunk_size]
            if len(chunk) > 50:
                chunks.append(chunk)

print(f"分块后总数: {len(chunks)}")

print("生成 embeddings...")
batch_size = 256
all_embeddings = []
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    embs = embed_model.encode(batch)
    all_embeddings.append(embs)
    print(f"  已处理 {min(i+batch_size, len(chunks))}/{len(chunks)}")

all_embeddings = np.vstack(all_embeddings).astype("float32")
print(f"Embedding shape: {all_embeddings.shape}")

dim = all_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
faiss.normalize_L2(all_embeddings)
index.add(all_embeddings)
print(f"FAISS 索引已构建，共 {index.ntotal} 条")

faiss.write_index(index, "/root/autodl-tmp/data/law_faiss.index")
with open("/root/autodl-tmp/data/law_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("知识库构建完成！")