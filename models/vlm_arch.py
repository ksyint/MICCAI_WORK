import torch
import torch.nn as nn
from .encoder.builder import build_vision_tower
from .projector.builder import build_mm_projector
from .attention.ssa import SSABridge
from .attention.mssa import MSSAModule


class VLMMetaModel:
    def __init__(self, config):
        super(VLMMetaModel, self).__init__(config)
        if hasattr(config, "vision_tower"):
            self.vision_tower = build_vision_tower(config)
            self.mm_projector = build_mm_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if isinstance(vision_tower, list):
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args):
        self.config.input_size = model_args.input_size
        self.config.patch_size = model_args.patch_size
        self.config.dim = model_args.dim
        self.config.depth = model_args.depth
        self.config.vision_tower = model_args.vision_tower
        self.config.vision_select_layer = model_args.vision_select_layer
        self.config.vision_select_feature = model_args.vision_select_feature
        self.config.mm_projector_type = model_args.mm_projector_type
        self.config.mm_mlp_depth = model_args.mm_mlp_depth
        self.config.proj_out_num = model_args.proj_out_num

        if self.get_vision_tower() is None:
            self.vision_tower = build_vision_tower(self.config)
            self.vision_tower.requires_grad_(not model_args.freeze_vision_tower)
            if self.config.vision_tower == "dcformer":
                self.config.low_input_size = self.vision_tower.low_input_size
                self.config.high_input_size = self.vision_tower.high_input_size
            elif self.config.mm_projector_type == "mixer":
                self.config.low_output_size = model_args.low_output_size
                self.config.high_output_size = model_args.high_output_size
                self.config.low_input_size = (256, 384)
                self.config.high_input_size = (32, 768)

        if model_args.pretrain_vision_model is not None:
            weights = torch.load(model_args.pretrain_vision_model, map_location="cpu")
            self.vision_tower.vision_tower.load_state_dict(weights, strict=True)

        self.config.mm_hidden_size = self.vision_tower.hidden_size

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_mm_projector(self.config)

        if model_args.pretrain_mm_mlp_adapter is not None:
            adapter_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
            filtered = {k.split("mm_projector.")[-1]: v for k, v in adapter_weights.items()
                        if "mm_projector" in k}
            self.mm_projector.load_state_dict(filtered, strict=False)
