import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import monai.transforms as mtf
from prompt.visual_prompt import generate_prompted_nifti
from prompt.text_rewriter import rewrite_question, SYSTEM_PROMPT


class RadGenomeDataset(Dataset):
    def __init__(self, data_root, json_path, tokenizer, split="train",
                 input_size=(256, 256, 128), max_length=512,
                 use_visual_prompt=False, mask_dir=None,
                 prompt_color="blue", prompt_thickness=3):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_visual_prompt = use_visual_prompt
        self.mask_dir = mask_dir
        self.prompt_color = prompt_color
        self.prompt_thickness = prompt_thickness
        self.input_size = input_size
        with open(json_path, "r") as f:
            all_data = json.load(f)
        if isinstance(all_data, dict) and split in all_data:
            self.samples = all_data[split]
        elif isinstance(all_data, list):
            n = len(all_data)
            if split == "train":
                self.samples = all_data[:int(0.7 * n)]
            elif split == "val":
                self.samples = all_data[int(0.7 * n):int(0.8 * n)]
            else:
                self.samples = all_data[int(0.8 * n):]
        else:
            self.samples = all_data
        self.resize = mtf.Compose([
            mtf.EnsureChannelFirst(channel_dim="no_channel"),
            mtf.Resize(spatial_size=list(input_size), mode="bilinear"),
        ])

    def __len__(self):
        return len(self.samples)

    def _load_volume(self, nifti_path):
        full_path = os.path.join(self.data_root, nifti_path)
        if os.path.exists(full_path):
            img = sitk.ReadImage(full_path)
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
        else:
            arr = np.random.randn(*self.input_size).astype(np.float32)
        arr = arr - arr.min()
        arr = arr / max(arr.max(), 1e-8)
        tensor = torch.from_numpy(arr).unsqueeze(0).float()
        resized = self.resize(tensor)
        return resized

    def _load_mask(self, mask_path):
        full_path = os.path.join(self.data_root, mask_path)
        if os.path.exists(full_path):
            img = sitk.ReadImage(full_path)
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            tensor = torch.from_numpy(arr).unsqueeze(0).float()
            resize_mask = mtf.Resize(spatial_size=list(self.input_size), mode="nearest")
            resized = resize_mask(tensor)
            return (resized > 0.5).float()
        return None

    def __getitem__(self, idx):
        sample = self.samples[idx]
        nifti_path = sample.get("nifti_path", sample.get("image_path", ""))
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        is_closed = sample.get("is_closed", sample.get("answer_type", "open") == "closed")
        volume = self._load_volume(nifti_path)
        if self.use_visual_prompt and self.mask_dir:
            mask_path = sample.get("mask_path", None)
            if mask_path:
                mask = self._load_mask(mask_path)
                if mask is not None:
                    vol_np = volume.squeeze(0).numpy()
                    mask_np = mask.squeeze(0).numpy()
                    prompted, _ = generate_prompted_nifti(
                        vol_np, mask_np, self.prompt_color, self.prompt_thickness)
                    volume = torch.from_numpy(prompted).unsqueeze(0).float()
                    vol_max = volume.max()
                    if vol_max > 0:
                        volume = volume / vol_max
            question = rewrite_question(question, self.prompt_color)
        image_tokens = "<im_patch>" * 256
        input_text = image_tokens + question
        encoded = self.tokenizer(
            input_text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        if is_closed:
            label_text = answer
        else:
            label_text = answer
        label_encoded = self.tokenizer(
            label_text, max_length=self.max_length, padding="max_length",
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
