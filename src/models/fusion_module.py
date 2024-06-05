import torch
import numpy as np
from typing import List, Union, Dict

from .models import create_model
from .utils import Lambda

POOL_FUNC_NAMES = ["sum", "avg","mean","linearsum", "prod", "mul" ,"max", "pool", "weightedsum"]
STACK_FUNC_NAMES = ["concat" ,"stack", "concatenate", "stacking", "concatenating", "cat"]

#only pooling funcs
class LinearSum_(torch.nn.Module):
    def forward(self, x): return torch.sum(x, dim=-1)

class UniformSum_(torch.nn.Module):
    def forward(self, x): return torch.sum(x, dim=-1) / x.shape[-1]

class Product_(torch.nn.Module):
    def forward(self, x): return torch.prod(x, dim=-1)

class Maximum_(torch.nn.Module):
    def forward(self, x): return torch.max(x, dim=-1)[0]

class Stacking_(torch.nn.Module):
    def forward(self, x): return torch.stack(x, dim=-1)

class Concatenate_(torch.nn.Module):
    def forward(self, x): return torch.cat(x, dim=-1)

class WeightedSum_Vector(torch.nn.Module):
    def __init__(self, N_views, normalized=False, views_normalized = False, **kwargs):
        """
            if normalized, then weights applied are individuallly in [0,1]
            if views_normalized, then views applied have to sum 1, \sum_v alphas_v = 1
        """
        super(WeightedSum_Vector, self).__init__()
        self.N_views = N_views
        self.fusion_weights = torch.nn.Parameter(data=torch.Tensor(self.N_views), requires_grad=True)
        self.fusion_weights.data.uniform_(-1, 1)
        self.normalized = normalized
        self.views_normalized = views_normalized

    def forward(self, x):
        alphas = self.get_weights()
        return torch.tensordot(x, alphas, dims=[ [-1], [0]] )

    def get_weights(self):
        alphas = self.fusion_weights #un-normalized weights
        if self.normalized:
            alphas = torch.sigmoid(alphas)
        elif self.views_normalized:
            alphas = torch.nn.Softmax(dim=-1)(alphas)
        return alphas.squeeze()

class FusionModule(torch.nn.Module):
    def __init__(self, emb_dims: List[int], mode: str, adaptive: bool=False, features: bool=False, activation_fun="softmax",**kwargs):
        super(FusionModule, self).__init__()
        self.mode = mode
        self.adaptive = adaptive
        self.features = features #only used when adaptive=True
        self.activation_fun = activation_fun #only used when adaptive=True
        if type(emb_dims) == dict: #assuming a orderer list based on dictionary
            emb_dims = list(emb_dims.values())
        self.emb_dims = emb_dims
        self.N_views = len(emb_dims)
        self.joint_dim, self.feature_pool = self.get_dim_agg()
        self.check_valid_args()

        if self.feature_pool:
            self.stacker_function = Stacking_()

        if self.mode in STACK_FUNC_NAMES:
            self.concater_function = Concatenate_()

        elif self.mode in ["avg","mean","uniformsum"]:
            self.pooler_function = UniformSum_()

        elif self.mode in ["sum","add","linearsum"]:
            self.pooler_function = LinearSum_()

        elif self.mode in ["prod", "mul"]:
            self.pooler_function = Product_()

        elif self.mode in ["max", "pool"]:
            self.pooler_function = Maximum_()
        elif self.mode in ["weightedsum"]:
            self.pooler_function = WeightedSum_Vector(self.N_views, **kwargs)

        elif self.mode in ["location"] and self.adaptive:
            pass
        else:
            raise ValueError(f'Invalid value for mode: {self.mode}. Valid values: {POOL_FUNC_NAMES+STACK_FUNC_NAMES}')

        if self.adaptive:
            if self.mode in STACK_FUNC_NAMES:
                forward_input_dim = sum(self.emb_dims)
            else:
                forward_input_dim = self.joint_dim
            out_probs = self.N_views if self.mode not in ["location"] else 1
            forward_output_dim = self.joint_dim*out_probs if self.features else out_probs

            if "adaptive_args" in kwargs:
                self.attention_function = create_model(forward_input_dim, forward_output_dim, layer_size=forward_input_dim, **kwargs["adaptive_args"])
            else:
                self.attention_function = torch.nn.Linear(forward_input_dim, forward_output_dim)

        if "additional_layers" in kwargs:
            self.additional_layers = kwargs["additional_layers"]

    def get_dim_agg(self):
        if self.adaptive or (self.mode not in STACK_FUNC_NAMES):
            fusion_dim = self.emb_dims[0]
            feature_pool = True
        else:
            fusion_dim = sum(self.emb_dims)
            feature_pool = False
        return fusion_dim, feature_pool

    def check_valid_args(self):
        if len(np.unique(self.emb_dims)) != 1:
            if self.adaptive:
                raise Exception("Cannot set adaptive=True when the number of features in embedding are not the same")
            if self.mode in POOL_FUNC_NAMES:
                raise Exception("Cannot set pooling aggregation when the number of features in embedding are not the same")


    def forward(self, views_emb: List[torch.Tensor]) -> Dict[str, torch.Tensor]: #the list is always orderer based on previous models
        if self.feature_pool:
            views_stacked = self.stacker_function(views_emb)

        if self.mode in STACK_FUNC_NAMES:
            joint_emb_views = self.concater_function(views_emb)

        elif self.mode in POOL_FUNC_NAMES:
            joint_emb_views = self.pooler_function(views_stacked)
            if self.mode == "weightedsum":
                att_views = self.pooler_function.get_weights().repeat(len(joint_emb_views), 1)

        if self.adaptive:
            if  self.mode in ["location"]:
                att_views = torch.concat([self.attention_function(v) for v in views_emb], dim=1)
            else:
                att_views = self.attention_function(joint_emb_views)
            if self.features:
                att_views = torch.reshape(att_views, (att_views.size()[0],self.joint_dim,self.N_views))
            else:
                att_views = att_views[:,None,:]

            if self.activation_fun.lower() == "softmax":
                att_views = torch.nn.Softmax(dim=-1)(att_views)
            elif self.activation_fun.lower() == "tanh":
                att_views = torch.nn.Tanh()(att_views)
            elif self.activation_fun.lower() == "sigmoid":
                att_views = torch.nn.Sigmoid()(att_views)
            joint_emb_views = torch.sum(views_stacked*att_views, dim=-1)

        dic_return = {"joint_rep": joint_emb_views}
        if self.adaptive or self.mode == "weightedsum":
            dic_return["att_views"] = att_views
        return dic_return

    def get_info_dims(self):
        return { "emb_dims":self.emb_dims, "joint_dim":self.joint_dim, "feature_pool": self.feature_pool}

    def get_joint_dim(self):
        return self.joint_dim
