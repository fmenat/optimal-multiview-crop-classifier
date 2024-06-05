import torch
from torch import nn
import abc

class Base_Encoder(abc.ABC, nn.Module):
    """
    Class to add methods for common modality specific methods
    """

    @abc.abstractmethod
    def get_output_size(self):
        pass


class Generic_Encoder(Base_Encoder):
    """
        it adds a linear layer at the end of an encoder model with possible batch normalization.
        The linear layer could be variational in some extension
    """
    def __init__(
        self,
        encoder: nn.Module,
        latent_dims: int,
        use_norm: bool = False,
        variational: bool = False,
        **kwargs,
    ):
        super(Generic_Encoder, self).__init__()
        self.return_all = False
        self.pre_encoder = encoder

        #build encoder head
        self.latent_dims = latent_dims
        self.use_norm = use_norm
        self.linear_layer = nn.Linear(self.pre_encoder.get_output_size(), self.latent_dims)
        self.bn_linear = nn.BatchNorm1d(self.latent_dims, affine=False)
        #self.ln_linear = nn.LayerNorm(self.latent_dims, elementwise_affine=False)

    def activate_return_all(self):
        self.return_all = True

    def activate_normalize_output(self):
        self.use_norm = True

    def forward(self, x):
        out_forward = self.pre_encoder(x) #should return a dictionary with output data {"rep": tensor}, or a single tensor
        if type(out_forward) != dict:
            out_forward = {"rep": out_forward}

        return_dic = {"rep": self.linear_layer(out_forward["rep"])}
        if self.use_norm:
            return_dic["rep"] = self.bn_linear(return_dic["rep"])
            #return_dic["rep"] = self.ln_linear(return_dic["rep"])

        if self.return_all:
            return_dic["pre:rep"] = out_forward.pop("rep")
            return dict(**return_dic, **out_forward)
        else:
            return return_dic["rep"] #single tensor output

    def get_output_size(self):
        return self.latent_dims
