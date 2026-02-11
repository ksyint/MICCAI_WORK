import os
import json
import torch
import numpy as np
from torch.cuda.amp import autocast
from tqdm import tqdm
from utils.metrics import compute_bleu, compute_rouge, compute_meteor, compute_accuracy


class VLMEvaluator:
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if args.bf16 else torch.float32

    @torch.no_grad()
    def evaluate(self, dataloader, split="test"):
        self.model.to(self.device)
        self.model.eval()
        all_preds = []
        all_refs = []
        all_questions = []
        closed_preds = []
        closed_refs = []
        for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
            images = batch["images"].to(self.device, dtype=self.dtype)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            is_closed_list = batch["is_closed"]
            with autocast(dtype=self.dtype, enabled=self.args.bf16):
                generated = self.model.generate(
                    input_ids=input_ids, images=images,
                    attention_mask=attention_mask,
                    max_new_tokens=getattr(self.args, "max_new_tokens", 256),
                    do_sample=False, num_beams=1)
            for i in range(generated.shape[0]):
                pred_text = self.tokenizer.decode(
                    generated[i], skip_special_tokens=True).strip()
                ref_text = batch["answer"][i] if "answer" in batch else ""
                question = batch["question"][i] if "question" in batch else ""
                all_preds.append(pred_text)
                all_refs.append(ref_text)
                all_questions.append(question)
                if i < len(is_closed_list) and is_closed_list[i]:
                    closed_preds.append(pred_text)
                    closed_refs.append(ref_text)
        metrics = {}
        if all_preds and all_refs:
            metrics["bleu4"] = compute_bleu(all_preds, all_refs)
            metrics["rouge_l"] = compute_rouge(all_preds, all_refs)
            metrics["meteor"] = compute_meteor(all_preds, all_refs)
        if closed_preds and closed_refs:
            metrics["closed_accuracy"] = compute_accuracy(closed_preds, closed_refs)
        return metrics, all_preds, all_refs

    def save_results(self, metrics, preds, refs, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results = {
            "metrics": metrics,
            "predictions": [
                {"pred": p, "ref": r} for p, r in zip(preds, refs)
            ],
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
