# For fine-tuned models:
```
python roberta_finetune.py --epochs 20 --model roberta-base 
```

# For LLM:
```
python LLM_inference.py --model meta-llama/Llama-3.1-8B-Instruct # (This needs ~23GB VRAM without quantization)
```

# Accuracy Records

## Fine-tuned (best yet marked in bold)

| Method      | Plain | Data Augmentation | Pairwise training | Question to answer cross-attention |
| ----------- | ----------- |-----------|-----------|-----------
| RoBERTa    | 0.57       | |||
| RoBERTa-large, baseline   | **0.76**        ||||
| sentence transformer   |       ||||
| xlnet-base-uncased   |        ||||
| Deberta-v3-base  |        ||||



| Method      | Best Accuracy | Description |
| ----------- | ----------- |-----------
| RoBERTa-base, baseline    | 0.57       | |
| RoBERTa-large, baseline   | **0.76**        ||


## LLM (best yet marked in bold)
| Method      | Best Accuracy | Description |
| ----------- | ----------- |-----------
| Llama-3.1 8B    |0.86      | |
