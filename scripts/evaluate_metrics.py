import json
import jieba
from collections import Counter
import numpy as np
from bert_score import score as bert_score

with open("/root/autodl-tmp/data/inference_results.json", "r") as f:
    results = json.load(f)

def compute_bleu4(reference, hypothesis):
    ref_tokens = list(jieba.cut(reference))
    hyp_tokens = list(jieba.cut(hypothesis))
    if len(hyp_tokens) == 0:
        return 0.0
    scores = []
    for n in range(1, 5):
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
        hyp_ngrams = Counter([tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1)])
        clipped = sum(min(hyp_ngrams[ng], ref_ngrams.get(ng, 0)) for ng in hyp_ngrams)
        total = sum(hyp_ngrams.values())
        if total == 0:
            scores.append(0)
        else:
            scores.append(clipped / total)
    if any(s == 0 for s in scores):
        return 0.0
    bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
    log_avg = sum(np.log(s) for s in scores) / 4
    return bp * np.exp(log_avg)

def compute_rouge_l_chinese(reference, hypothesis):
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if len(ref_chars) == 0 or len(hyp_chars) == 0:
        return 0.0
    m, n = len(ref_chars), len(hyp_chars)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[m][n]
    precision = lcs_len / n if n > 0 else 0
    recall = lcs_len / m if m > 0 else 0
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1

print("=" * 60)
print("评测结果")
print("=" * 60)

for config_name, items in results.items():
    rouge_scores = []
    bleu_scores = []
    for item in items:
        ref = item["reference"]
        hyp = item["answer"]
        rouge_scores.append(compute_rouge_l_chinese(ref, hyp))
        bleu_scores.append(compute_bleu4(ref, hyp))
    avg_rouge = np.mean(rouge_scores) * 100
    avg_bleu = np.mean(bleu_scores) * 100
    print(f"\n{config_name}:")
    print(f"  ROUGE-L: {avg_rouge:.1f}%")
    print(f"  BLEU-4:  {avg_bleu:.1f}%")

print("\n" + "=" * 60)

print("\n计算 BERTScore ...\n")

for config_name, items in results.items():
    refs = [item["reference"] for item in items]
    hyps = [item["answer"] for item in items]
    
    P, R, F1 = bert_score(
        hyps, refs,
        lang="zh",
        model_type="bert-base-chinese",
        verbose=False
    )
    
    print(f"{config_name}:")
    print(f"  BERTScore F1: {F1.mean().item() * 100:.1f}%")

print("\n" + "=" * 60)