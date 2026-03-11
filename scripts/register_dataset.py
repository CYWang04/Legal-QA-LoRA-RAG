import json

info_path = "/root/autodl-tmp/LLaMA-Factory/data/dataset_info.json"
with open(info_path, "r") as f:
    dataset_info = json.load(f)

dataset_info["disc_law_train"] = {
    "file_name": "disc_law_train.json",
    "formatting": "alpaca"
}

with open(info_path, "w") as f:
    json.dump(dataset_info, f, indent=2, ensure_ascii=False)

print("数据集已注册")