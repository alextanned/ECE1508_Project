# For fine-tuned models:
```
python roberta_finetune.py --epochs 20 --model roberta-base 
```

# For LLM:
```
python LLM_inference.py --model meta-llama/Llama-3.1-8B-Instruct # (This needs ~23GB VRAM without quantization)
```
