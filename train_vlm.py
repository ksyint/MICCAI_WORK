import os
import sys
import torch
from torch.utils.data import DataLoader

from configs.default_config import get_train_args
from models.med3d_vlm import Med3DVLM, Med3DVLMConfig
from models.decoder.qwen_decoder import (
    load_qwen_decoder, apply_lora_to_decoder, freeze_decoder_base,
)
from data.radgenome_dataset import RadGenomeDataset
from data.pmcvqa_dataset import PMCVQADataset
from data.collator import VLMDataCollator
from losses.combined_loss import CombinedVQALoss
from engine.trainer import VLMTrainer
from utils.misc import seed_everything, print_model_info, get_lr_scheduler
from utils.logger import TrainingLogger
from utils.checkpoint import find_latest_checkpoint


def build_model(args):
    decoder, tokenizer, decoder_config = load_qwen_decoder(
        model_size=args.decoder_size,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32)
    num_new_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<im_patch>"]})
    decoder.resize_token_embeddings(len(tokenizer))
    if args.lora_enable:
        decoder = apply_lora_to_decoder(
            decoder, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout)
    vlm_config = Med3DVLMConfig(
        input_size=tuple(args.input_size),
        patch_size=tuple(args.input_size),
        dim=args.dim,
        depth=args.depth,
        vision_tower=args.vision_tower,
        vision_select_layer=args.vision_select_layer,
        vision_select_feature=args.vision_select_feature,
        mm_projector_type=args.mm_projector_type,
        mm_mlp_depth=args.mm_mlp_depth,
        mm_hidden_size=args.dim,
        hidden_size=decoder_config.hidden_size,
        proj_out_num=args.proj_out_num,
        use_ssa=args.use_ssa,
        use_mssa=args.use_mssa,
        ssa_layers=args.ssa_layers,
        mssa_layers=args.mssa_layers,
        mssa_heads=args.mssa_heads,
        max_slices=args.max_slices,
    )
    model = Med3DVLM(vlm_config, decoder, tokenizer)
    if args.freeze_vision:
        model.freeze_vision()
    if args.pretrain_vision_model:
        weights = torch.load(args.pretrain_vision_model, map_location="cpu")
        model.vision_tower.vision_tower.load_state_dict(weights, strict=True)
    if args.pretrain_mm_adapter:
        adapter_weights = torch.load(args.pretrain_mm_adapter, map_location="cpu")
        filtered = {}
        for k, v in adapter_weights.items():
            if "mm_projector" in k:
                new_k = k.split("mm_projector.")[-1]
                filtered[new_k] = v
        model.mm_projector.load_state_dict(filtered, strict=False)
    if args.pretrain_mllm:
        state = torch.load(args.pretrain_mllm, map_location="cpu")
        model.load_state_dict(state, strict=False)
    return model, tokenizer


def build_datasets(args, tokenizer):
    if args.dataset == "radgenome":
        train_dataset = RadGenomeDataset(
            data_root=args.data_root, json_path=args.radgenome_json,
            tokenizer=tokenizer, split="train",
            input_size=tuple(args.input_size), max_length=args.max_length,
            use_visual_prompt=args.use_visual_prompt, mask_dir=args.mask_dir,
            prompt_color=args.prompt_color, prompt_thickness=args.prompt_thickness)
        val_dataset = RadGenomeDataset(
            data_root=args.data_root, json_path=args.radgenome_json,
            tokenizer=tokenizer, split="val",
            input_size=tuple(args.input_size), max_length=args.max_length,
            use_visual_prompt=args.use_visual_prompt, mask_dir=args.mask_dir,
            prompt_color=args.prompt_color, prompt_thickness=args.prompt_thickness)
    elif args.dataset == "pmcvqa":
        train_dataset = PMCVQADataset(
            data_root=args.data_root, json_path=args.pmcvqa_json,
            tokenizer=tokenizer, split="train",
            input_size=tuple(args.input_size), max_length=args.max_length,
            use_visual_prompt=args.use_visual_prompt,
            prompt_color=args.prompt_color)
        val_dataset = PMCVQADataset(
            data_root=args.data_root, json_path=args.pmcvqa_json,
            tokenizer=tokenizer, split="val",
            input_size=tuple(args.input_size), max_length=args.max_length,
            use_visual_prompt=args.use_visual_prompt,
            prompt_color=args.prompt_color)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return train_dataset, val_dataset


def main():
    args = get_train_args()
    seed_everything(args.seed)
    logger = TrainingLogger(log_dir=args.log_dir)
    logger.info(f"Args: {vars(args)}")
    model, tokenizer = build_model(args)
    print_model_info(model, name="Med3DVLM")
    print_model_info(model.decoder, name="Decoder")
    train_dataset, val_dataset = build_datasets(args, tokenizer)
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    collator = VLMDataCollator()
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collator)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collator)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.98))
    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    scheduler = get_lr_scheduler(
        optimizer, scheduler_type="cosine_warmup",
        num_training_steps=total_steps, warmup_steps=args.warmup_steps)
    loss_fn = CombinedVQALoss(
        focal_gamma=args.focal_gamma, lambda_cls=args.lambda_cls,
        lambda_gen=args.lambda_gen, lambda_reg=args.lambda_reg)
    trainer = VLMTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        loss_fn=loss_fn, args=args,
        train_loader=train_loader, val_loader=val_loader,
        writer=logger.writer)
    if args.resume_from:
        ckpt_path = args.resume_from
    else:
        ckpt_path = find_latest_checkpoint(args.output_dir)
    if ckpt_path and os.path.exists(ckpt_path):
        logger.info(f"Resuming from: {ckpt_path}")
        trainer.load_checkpoint(ckpt_path)
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")
    logger.close()


if __name__ == "__main__":
    main()
