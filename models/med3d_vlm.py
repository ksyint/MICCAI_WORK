import torch
import torch.nn as nn
from .encoder.builder import build_vision_tower
from .projector.builder import build_mm_projector
from .attention.ssa import SSABridge
from .attention.mssa import MSSAModule


class Med3DVLMConfig:
    def __init__(self, **kwargs):
        self.input_size = kwargs.get("input_size", (256, 256, 128))
        self.patch_size = kwargs.get("patch_size", (16, 16, 16))
        self.dim = kwargs.get("dim", 768)
        self.depth = kwargs.get("depth", 12)
        self.vision_tower = kwargs.get("vision_tower", "dcformer")
        self.vision_select_layer = kwargs.get("vision_select_layer", -2)
        self.vision_select_feature = kwargs.get("vision_select_feature", "cls_patch")
        self.mm_projector_type = kwargs.get("mm_projector_type", "mixer")
        self.mm_mlp_depth = kwargs.get("mm_mlp_depth", 2)
        self.mm_hidden_size = kwargs.get("mm_hidden_size", 768)
        self.hidden_size = kwargs.get("hidden_size", 1536)
        self.proj_out_num = kwargs.get("proj_out_num", 256)
        self.low_input_size = kwargs.get("low_input_size", (256, 384))
        self.high_input_size = kwargs.get("high_input_size", (32, 768))
        self.low_output_size = kwargs.get("low_output_size", [192, 128])
        self.high_output_size = kwargs.get("high_output_size", [64, 128])
        self.use_ssa = kwargs.get("use_ssa", True)
        self.use_mssa = kwargs.get("use_mssa", True)
        self.ssa_layers = kwargs.get("ssa_layers", 2)
        self.mssa_layers = kwargs.get("mssa_layers", 2)
        self.mssa_heads = kwargs.get("mssa_heads", 4)
        self.max_slices = kwargs.get("max_slices", 512)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)


class Med3DVLM(nn.Module):
    def __init__(self, config, decoder, tokenizer):
        super().__init__()
        self.config = config
        self.vision_tower = build_vision_tower(config)
        self.mm_projector = build_mm_projector(config)
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.proj_out_num = config.proj_out_num
        decoder_hidden = self._get_decoder_hidden_size()
        if config.use_ssa:
            self.ssa_bridge = SSABridge(
                dim=decoder_hidden, num_layers=config.ssa_layers,
                num_heads=config.mssa_heads)
        else:
            self.ssa_bridge = None
        if config.use_mssa:
            self.mssa_module = MSSAModule(
                dim=decoder_hidden, num_layers=config.mssa_layers,
                num_heads=config.mssa_heads, max_slices=config.max_slices)
        else:
            self.mssa_module = None
        self.vision_proj = None
        vision_out_dim = self._compute_vision_out_dim()
        if vision_out_dim != decoder_hidden:
            self.vision_proj = nn.Linear(vision_out_dim, decoder_hidden)

    def _get_decoder_hidden_size(self):
        if hasattr(self.decoder, "config"):
            return self.decoder.config.hidden_size
        return self.config.hidden_size

    def _compute_vision_out_dim(self):
        if hasattr(self.mm_projector, "proj_out_num"):
            return self.config.hidden_size
        return self.config.mm_hidden_size

    def freeze_vision(self):
        self.vision_tower.requires_grad_(False)
        self.mm_projector.requires_grad_(False)
        if self.vision_proj is not None:
            self.vision_proj.requires_grad_(False)

    def encode_images(self, images):
        with torch.no_grad():
            image_features = self.vision_tower(images)
            image_features = self.mm_projector(image_features)
        return image_features

    def prepare_multimodal_inputs(self, input_ids, images, attention_mask=None):
        image_features = self.encode_images(images)
        if self.vision_proj is not None:
            image_features = self.vision_proj(image_features)
        embed_tokens = self.decoder.get_input_embeddings()
        text_embeds = embed_tokens(input_ids)
        if self.ssa_bridge is not None:
            image_features = self.ssa_bridge(image_features, text_embeds)
        if self.mssa_module is not None:
            image_features, text_embeds = self.mssa_module(image_features, text_embeds)
        inputs_embeds = torch.cat([
            text_embeds[:, :1, :],
            image_features,
            text_embeds[:, 1:, :],
        ], dim=1)
        if attention_mask is not None:
            img_mask = torch.ones(
                image_features.shape[0], image_features.shape[1],
                device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([
                attention_mask[:, :1], img_mask, attention_mask[:, 1:]
            ], dim=1)
        return inputs_embeds, attention_mask

    def forward(self, input_ids, images, attention_mask=None, labels=None, **kwargs):
        inputs_embeds, attention_mask = self.prepare_multimodal_inputs(
            input_ids, images, attention_mask)
        if labels is not None:
            img_len = inputs_embeds.shape[1] - input_ids.shape[1]
            ignore_labels = torch.full(
                (labels.shape[0], img_len), -100,
                device=labels.device, dtype=labels.dtype)
            labels = torch.cat([labels[:, :1], ignore_labels, labels[:, 1:]], dim=1)
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        return outputs

    @torch.no_grad()
    def generate(self, input_ids, images, attention_mask=None, **kwargs):
        inputs_embeds, attention_mask = self.prepare_multimodal_inputs(
            input_ids, images, attention_mask)
        outputs = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
        return outputs

    def get_trainable_params(self):
        params = []
        if self.ssa_bridge is not None:
            params.extend(self.ssa_bridge.parameters())
        if self.mssa_module is not None:
            params.extend(self.mssa_module.parameters())
        for name, param in self.decoder.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params
