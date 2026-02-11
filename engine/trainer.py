import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class VLMTrainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, args,
                 train_loader, val_loader=None, writer=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = writer or SummaryWriter(log_dir=args.log_dir)
        self.scaler = GradScaler(enabled=args.bf16)
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.dtype = torch.bfloat16 if args.bf16 else torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        self.model.to(self.device)
        self.model.train()
        for epoch in range(self.args.epochs):
            epoch_loss = self._train_epoch(epoch)
            self.writer.add_scalar("epoch/train_loss", epoch_loss, epoch)
            if self.val_loader is not None:
                val_loss = self._validate(epoch)
                self.writer.add_scalar("epoch/val_loss", val_loss, epoch)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)
            if (epoch + 1) % max(1, self.args.epochs // self.args.save_total_limit) == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        self.optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            images = batch["images"].to(self.device, dtype=self.dtype)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            with autocast(dtype=self.dtype, enabled=self.args.bf16):
                outputs = self.model(
                    input_ids=input_ids, images=images,
                    attention_mask=attention_mask, labels=labels)
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    loss = outputs.loss
                else:
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    lora_params = self.model.get_trainable_params() if hasattr(
                        self.model, "get_trainable_params") else None
                    loss, loss_dict = self.loss_fn(
                        gen_logits=shift_logits, gen_targets=shift_labels,
                        lora_params=lora_params)
            loss_scaled = loss / self.args.grad_accum_steps
            self.scaler.scale(loss_scaled).backward()
            if (step + 1) % self.args.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.global_step += 1
            total_loss += loss.item()
            num_batches += 1
            if self.global_step % self.args.log_steps == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                self.writer.add_scalar("train/lr", lr, self.global_step)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
            if self.args.eval_steps > 0 and self.global_step % self.args.eval_steps == 0:
                if self.val_loader is not None:
                    val_loss = self._validate(epoch)
                    self.writer.add_scalar("val/loss", val_loss, self.global_step)
                    self.model.train()
        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch["images"].to(self.device, dtype=self.dtype)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            with autocast(dtype=self.dtype, enabled=self.args.bf16):
                outputs = self.model(
                    input_ids=input_ids, images=images,
                    attention_mask=attention_mask, labels=labels)
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    loss = outputs.loss
                else:
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss, _ = self.loss_fn(
                        gen_logits=shift_logits, gen_targets=shift_labels)
            total_loss += loss.item()
            num_batches += 1
        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, epoch, is_best=False):
        os.makedirs(self.args.output_dir, exist_ok=True)
        if is_best:
            path = os.path.join(self.args.output_dir, "best_model.pt")
        else:
            path = os.path.join(self.args.output_dir, f"checkpoint-epoch{epoch}.pt")
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": {
                k: v for k, v in self.model.state_dict().items()
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        return ckpt.get("epoch", 0)
