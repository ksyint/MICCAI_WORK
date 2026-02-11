import os
import glob
import torch


def save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                    best_val_loss, output_dir, is_best=False):
    os.makedirs(output_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if is_best:
        path = os.path.join(output_dir, "best_model.pt")
    else:
        path = os.path.join(output_dir, f"checkpoint-{global_step}.pt")
    torch.save(state, path)
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return {
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", 0),
        "best_val_loss": ckpt.get("best_val_loss", float("inf")),
    }


def save_lora_weights(model, output_dir, filename="lora_weights.pt"):
    os.makedirs(output_dir, exist_ok=True)
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora" in name.lower() and param.requires_grad:
            lora_state[name] = param.data.cpu()
    path = os.path.join(output_dir, filename)
    torch.save(lora_state, path)
    return path


def load_lora_weights(model, path, device="cuda"):
    lora_state = torch.load(path, map_location=device)
    model_state = model.state_dict()
    for name, param in lora_state.items():
        if name in model_state:
            model_state[name].copy_(param)
    return model


def save_adapter_weights(model, output_dir, filename="adapter_weights.pt"):
    os.makedirs(output_dir, exist_ok=True)
    adapter_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            adapter_state[name] = param.data.cpu()
    path = os.path.join(output_dir, filename)
    torch.save(adapter_state, path)
    return path


def find_latest_checkpoint(output_dir):
    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*.pt"))
    if not ckpts:
        best = os.path.join(output_dir, "best_model.pt")
        return best if os.path.exists(best) else None
    ckpts.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
    return ckpts[-1]


def cleanup_checkpoints(output_dir, keep=2):
    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*.pt"))
    if len(ckpts) <= keep:
        return
    ckpts.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
    for ckpt in ckpts[:-keep]:
        os.remove(ckpt)
