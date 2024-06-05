from torch import nn
from typing import List, Union, Dict

from .base_fusion import MVFusion, MVFusionMultiLoss, SVPool, HybridFusion
from .fusion_module import FusionModule
from .utils import Lambda

class InputFusion(MVFusion):
    def __init__(self,
                 predictive_model,
                 fusion_module: dict = {},
                 loss_args: dict = {},
                 view_names: List[str] = [],
                 input_dim_to_stack: Union[List[int], Dict[str,int]] = 0,
                 ):
        if len(loss_args) == 0:
            loss_args = {"name": "mse"}
        if type(fusion_module) == dict:
            if len(fusion_module) == 0:
                fusion_module = {"mode": "concat", "adaptive":False, "emb_dims": input_dim_to_stack }
            fusion_module = FusionModule(**fusion_module)
        fake_view_encoders = [nn.Identity() for _ in range(fusion_module.N_views)]
        if "DEM" in view_names:
            for idx_, name_ in enumerate(view_names):
                if name_ == "DEM":
                    fake_view_encoders[idx_] = Lambda(lambda x: x.repeat(1,12,1))
        super(InputFusion, self).__init__(fake_view_encoders, fusion_module, predictive_model,
            loss_args=loss_args, view_names=view_names)

class DecisionFusion(MVFusion):
    def __init__(self,
                 view_encoders,
                 fusion_module: dict = {},
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 n_outputs: int = 0,
                 ):
        if len(loss_args) == 0:
            loss_args = {"name": "mse"}
        if type(fusion_module) == dict:
            if len(fusion_module) == 0:
                fusion_module = {"mode": "avg", "adaptive":False, "emb_dims":[n_outputs for _ in range(len(view_encoders))]}
            fusion_module = FusionModule(**fusion_module)
        super(DecisionFusion, self).__init__(view_encoders, fusion_module, nn.Identity(),
            loss_args=loss_args, view_names=view_names)
        self.n_outputs = n_outputs

class FeatureFusion(MVFusion):
    def __init__(self,
                 view_encoders,
                 fusion_module: nn.Module,
                 predictive_model: nn.Module,
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 ):
        if len(loss_args) == 0:
            loss_args = {"name": "mse"}
        super(FeatureFusion, self).__init__(view_encoders, fusion_module, predictive_model,
             loss_args=loss_args, view_names=view_names)

class FeatureFusionMultiLoss(MVFusionMultiLoss): #same asFeatureFusion
    def __init__(self,
                 view_encoders,
                 fusion_module: nn.Module,
                 predictive_model: nn.Module,
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 ):
        if len(loss_args) == 0:
            loss_args = {"name": "mse"}
        super(FeatureFusionMultiLoss, self).__init__(view_encoders, fusion_module, predictive_model,
             loss_args=loss_args, view_names=view_names)

class DecisionFusionMultiLoss(MVFusionMultiLoss):
    def __init__(self,
                 view_encoders,
                 fusion_module: dict = {},
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 n_outputs: int = 0,
                 ):
        if type(fusion_module) == dict:
            if len(fusion_module) == 0:
                fusion_module = {"mode": "avg", "adaptive":False, "emb_dims":[n_outputs for _ in range(len(view_encoders))]}
            fusion_module = FusionModule(**fusion_module)
        if len(loss_args) == 0:
            loss_args = {"name": "mse"}
        super(DecisionFusionMultiLoss, self).__init__(view_encoders, fusion_module, nn.Identity(),
            loss_args=loss_args, view_names=view_names)
        self.n_outputs = n_outputs


class HybridFusion_FD(HybridFusion):
    def __init__(self,
                 view_encoders,
                 fusion_module_feat: nn.Module,
                 predictive_model: nn.Module,
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 fusion_module_deci: nn.Module = None,
                 ):
        if len(loss_args) == 0:
            loss_args = {"name": "mse"}
        super(HybridFusion_FD, self).__init__(view_encoders, fusion_module_feat, predictive_model,
             loss_args=loss_args, view_names=view_names, fusion_module_deci=fusion_module_deci)

class SingleViewPool(SVPool):
    pass
