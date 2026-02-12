import os
import sys
import torch
from torch.utils.data import DataLoader

from configs.default_config import get_eval_args
from models.med3d_vlm import Med3DVLM, Med3DVLMConfig
from models.decoder.qwen_decoder import load_qwen_decoder, apply_lora_to_decoder
from data.radgenome_dataset import RadGenomeDataset
from data.pmcvqa_dataset import PMCVQADataset
from data.collator import VLMDataCollator
from engine.evaluator import VLMEvaluator
from utils.misc import seed_everything, print_model_info
from utils.checkpoint import load_checkpoint


def build_model(args):
    decoder, tokenizer, decoder_config = load_qwen_decoder(
        model_size=args.decoder_size,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<im_patch>"]})
    decoder.resize_token_embeddings(len(tokenizer))
    decoder = apply_lora_to_decoder(decoder)
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
    return model, tokenizer


def build_test_dataset(args, tokenizer):
    if args.dataset == "radgenome":
        dataset = RadGenomeDataset(
            data_root=args.data_root, json_path=args.radgenome_json,
            tokenizer=tokenizer, split="test",
            input_size=tuple(args.input_size), max_length=args.max_length,
            use_visual_prompt=args.use_visual_prompt, mask_dir=args.mask_dir,
            prompt_color=args.prompt_color, prompt_thickness=args.prompt_thickness)
    elif args.dataset == "pmcvqa":
        dataset = PMCVQADataset(
            data_root=args.data_root, json_path=args.pmcvqa_json,
            tokenizer=tokenizer, split="test",
            input_size=tuple(args.input_size), max_length=args.max_length,
            use_visual_prompt=args.use_visual_prompt,
            prompt_color=args.prompt_color)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return dataset


def main():
    args = get_eval_args()
    seed_everything(args.seed)
    model, tokenizer = build_model(args)
    ckpt_info = load_checkpoint(args.checkpoint_path, model)
    print(f"Loaded checkpoint from epoch {ckpt_info['epoch']}, "
          f"step {ckpt_info['global_step']}")
    print_model_info(model, name="Med3DVLM")
    test_dataset = build_test_dataset(args, tokenizer)
    print(f"Test samples: {len(test_dataset)}")
    collator = VLMDataCollator()
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collator)
    evaluator = VLMEvaluator(model, tokenizer, args)
    metrics, preds, refs = evaluator.evaluate(test_loader, split="test")
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")
    output_path = os.path.join(args.output_dir, f"eval_{args.dataset}.json")
    evaluator.save_results(metrics, preds, refs, output_path)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
