import json
import random
import os

random.seed(42)

pair_qa = []
with open("/root/autodl-tmp/data/DISC-Law-SFT-Pair-QA-released.jsonl", "r") as f:
    for line in f:
        item = json.loads(line.strip())
        pair_qa.append(item)

print(f"Pair-QA 总量: {len(pair_qa)}")
random.shuffle(pair_qa)

train_data = pair_qa[:5000]
test_data = pair_qa[5000:5200]
print(f"训练集: {len(train_data)}, 测试集: {len(test_data)}")

os.makedirs("/root/autodl-tmp/LLaMA-Factory/data", exist_ok=True)

alpaca_train = []
for item in train_data:
    alpaca_train.append({
        "instruction": item["input"],
        "input": "",
        "output": item["output"]
    })

with open("/root/autodl-tmp/LLaMA-Factory/data/disc_law_train.json", "w", encoding="utf-8") as f:
    json.dump(alpaca_train, f, ensure_ascii=False, indent=2)

with open("/root/autodl-tmp/data/test_set.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print("训练集已保存")
print("测试集已保存")

# 知识库
triplet_qa = []
with open("/root/autodl-tmp/data/DISC-Law-SFT-Triplet-QA-released.jsonl", "r") as f:
    for line in f:
        item = json.loads(line.strip())
        triplet_qa.append(item)

print(f"Triplet-QA 总量: {len(triplet_qa)}")

law_texts = set()
for item in triplet_qa:
    if "reference" in item and item["reference"]:
        ref = item["reference"]
        if isinstance(ref, list):
            ref = "\n".join(ref)
        ref = ref.strip()
        if len(ref) > 20:
            law_texts.add(ref)

law_texts = list(law_texts)
print(f"去重后法条文本数量: {len(law_texts)}")

with open("/root/autodl-tmp/data/law_knowledge_base.jsonl", "w", encoding="utf-8") as f:
    for i, text in enumerate(law_texts):
        json.dump({"id": i, "text": text}, f, ensure_ascii=False)
        f.write("\n")

print("知识库已保存")