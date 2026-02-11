import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import monai.transforms as mtf
from prompt.text_rewriter import rewrite_question


class PMCVQADataset(Dataset):
    def __init__(self, data_root, json_path, tokenizer, split="train",
                 input_size=(256, 256, 128), max_length=512,
                 use_visual_prompt=False, prompt_color="blue"):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_visual_prompt = use_visual_prompt
        self.prompt_color = prompt_color
        self.input_size = input_size
        with open(json_path, "r") as f:
            all_data = json.load(f)
        if isinstance(all_data, dict) and split in all_data:
            self.samples = all_data[split]
        elif isinstance(all_data, list):
            n = len(all_data)
            if split == "train":
                self.samples = all_data[:int(0.8 * n)]
            elif split == "val":
                self.samples = all_data[int(0.8 * n):int(0.9 * n)]
            else:
                self.samples = all_data[int(0.9 * n):]
        else:
            self.samples = all_data
        self.resize = mtf.Compose([
            mtf.EnsureChannelFirst(channel_dim="no_channel"),
            mtf.Resize(spatial_size=list(input_size), mode="bilinear"),
        ])

    def __len__(self):
        return len(self.samples)

    def _load_volume(self, path):
        full_path = os.path.join(self.data_root, path)
        if os.path.exists(full_path):
            if full_path.endswith((".nii", ".nii.gz")):
                img = sitk.ReadImage(full_path)
                arr = sitk.GetArrayFromImage(img).astype(np.float32)
            elif full_path.endswith(".npy"):
                arr = np.load(full_path).astype(np.float32)
            else:
                arr = np.random.randn(*self.input_size).astype(np.float32)
        else:
            arr = np.random.randn(*self.input_size).astype(np.float32)
        arr = arr - arr.min()
        arr = arr / max(arr.max(), 1e-8)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, 0).repeat(self.input_size[2], axis=0)
        tensor = torch.from_numpy(arr).unsqueeze(0).float()
        resized = self.resize(tensor)
        return resized

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample.get("image_path", sample.get("nifti_path", ""))
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        is_closed = sample.get("is_closed", False)
        volume = self._load_volume(image_path)
        if self.use_visual_prompt:
            question = rewrite_question(question, self.prompt_color)
        image_tokens = "<im_patch>" * 256
        input_text = image_tokens + question
        encoded = self.tokenizer(
            input_text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        label_encoded = self.tokenizer(
            answer, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt")
        labels = label_encoded["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "image": volume,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "is_closed": is_closed,
            "question": question,
            "answer": answer,
        }
