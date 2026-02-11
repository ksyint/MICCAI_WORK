import torch


class VLMDataCollator:
    def __call__(self, batch):
        images = torch.stack([b["image"] for b in batch], dim=0)
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        is_closed = [b["is_closed"] for b in batch]
        return {
            "images": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "is_closed": is_closed,
        }


class CLIPDataCollator:
    def __init__(self, gather_all=False):
        self.gather_all = gather_all

    def __call__(self, batch):
        images = torch.stack([b["image"] for b in batch], dim=0)
        input_ids = torch.stack([b["input_id"] for b in batch], dim=0)
        attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
        batch_size = images.shape[0]
        if self.gather_all:
            try:
                world_size = torch.distributed.get_world_size()
                batch_size *= world_size
            except Exception:
                pass
        labels = torch.arange(batch_size, device=images.device, dtype=torch.long)
        return {
            "images": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
