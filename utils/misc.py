import os
import random
import numpy as np
import torch
import torch.distributed as dist


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def print_model_info(model, name="Model"):
    total, trainable, frozen = count_parameters(model)
    ratio = trainable / max(total, 1) * 100
    print(f"{name}: Total={total:,} Trainable={trainable:,} "
          f"Frozen={frozen:,} Ratio={ratio:.2f}%")


def is_rank_zero():
    if "RANK" in os.environ:
        if int(os.environ["RANK"]) != 0:
            return False
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return False
    return True


def rank0_print(*args, **kwargs):
    if is_rank_zero():
        print(*args, **kwargs)


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_lr_scheduler(optimizer, scheduler_type, num_training_steps, warmup_steps=0):
    if scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=num_training_steps - warmup_steps)
    elif scheduler_type == "linear":
        from torch.optim.lr_scheduler import LinearLR
        return LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                        total_iters=num_training_steps)
    elif scheduler_type == "cosine_warmup":
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                          total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=num_training_steps - warmup_steps)
        return SequentialLR(optimizer, schedulers=[warmup, cosine],
                            milestones=[warmup_steps])
    return None
