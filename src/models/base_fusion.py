import torch, copy
from torch import nn
import pytorch_lightning as pl
import numpy as np
from typing import List, Union, Dict

from .utils import stack_all, object_to_list, collate_all_list, detach_all
from .utils import get_dic_emb_dims, get_loss_by_name, map_encoders, count_parameters
from .fusion_module import FusionModule, POOL_FUNC_NAMES

class _BaseViewsLightning(pl.LightningModule):
    def __init__(
            self,
            optimizer="adam",
            lr=1e-3,
            weight_decay=0,
            extra_optimizer_kwargs=None,
            lr_decay_steps=None,
    ):
        super().__init__()
        if extra_optimizer_kwargs is None:
            extra_optimizer_kwargs = {}
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.extra_optimizer_kwargs = extra_optimizer_kwargs

    def training_step(self, batch, batch_idx):
        """
            batch sould be a dictionary containin key 'views' for data and 'target' for the desired output to learn
        """
        loss = self.loss_batch(batch)
        for k, v in loss.items():
            self.log("train_" + k, v, prog_bar=True)
        return loss["objective"]

    def validation_step(self, batch, batch_idx):
        """
            batch sould be a dictionary containin key 'views' for data and 'target' for the desired output to learn
        """
        loss = self.loss_batch(batch)
        for k, v in loss.items():
            self.log("val_" + k, v)
        return loss["objective"]

    def configure_optimizers(self):
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.
        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW
        elif self.optimizer == "lbfgs":
            optimizer = torch.optim.LBFGS
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam, adamw)")
        return  optimizer(self.parameters(),lr=self.lr, weight_decay=self.weight_decay, **self.extra_optimizer_kwargs)

    def count_parameters(self):
        return count_parameters(self)

class MVFusion(_BaseViewsLightning):
    #ONLY FOR POINT-PREDICTION
    #it is based on three modules: encoders, aggregation, predictive_model
    #only one-task (prediction) and full-view available setting
    #support list and dictionary of encoders -- but transform to dict (ideally it should be always a dict)
    #support list and dictionary of emb dims -- but transform to dict
    def __init__(self,
                 view_encoders: Union[List[nn.Module],Dict[str,nn.Module]],  #require that it contain get_output_size() .. otherwise indicate in emb_dims..
                 fusion_module: nn.Module,
                 predictive_model: nn.Module,
                 loss_args: dict ={},
                 view_names: List[str] = [], #this is only used if view_encoders are a list
                 ):
        super(MVFusion, self).__init__()
        if len(view_encoders) == 0:
            raise Exception("you have to give a encoder models (nn.Module), currently view_encoders=[] or {}")
        if type(predictive_model) == type(None):
            raise Exception("you need to define a predictive_model")
        if type(fusion_module) == type(None):
            raise Exception("you need to define a fusion_module")
        if len(loss_args) == 0:
            loss_args = {"name": "mse"}
        self.save_hyperparameters(ignore=['view_encoders','predictive_model', 'fusion_module'])

        view_encoders = map_encoders(view_encoders, view_names=view_names) #view_encoders to dict if no dict yet (e.g. list)
        self.views_encoder = nn.ModuleDict(view_encoders)
        self.view_names = list(self.views_encoder.keys())
        self.fusion_module = fusion_module
        self.predictive_model = predictive_model

        self.criteria = loss_args["function"] if "function" in loss_args else get_loss_by_name(**self.hparams_initial.loss_args)

        self.set_additional()

    def set_additional(self):
        self.N_views = len(self.views_encoder)
        if type(self.predictive_model) == nn.Identity:
            self.where_fusion = "decision"
        elif all([nn.Identity == type(v) for v in self.views_encoder.values()]):
            self.where_fusion = "input"
        else:
            self.where_fusion = "feature"

        if hasattr(self.fusion_module, "get_info_dims"):
            info = self.fusion_module.get_info_dims()
            self.joint_dim = info["joint_dim"]
            self.feature_pool = info["feature_pool"]
        else:
            self.feature_pool = False
            self.joint_dim = 0
            self.where_fusion = "no_fusion_info"
        self.emb_dims = get_dic_emb_dims(self.views_encoder)

        if "decision" == self.where_fusion and (not self.feature_pool):
            raise Exception("Cannot use decision-level fusion with non feature_pools, perhaps trying set adaptive=True or use agg_mode=[mean, sum, max]")

    def apply_softmax(self, y: torch.Tensor) -> torch.Tensor:
        return nn.Softmax(dim=-1)(y)

    def forward_encoders(self,
            views: Dict[str, torch.Tensor],
            testing_views: list = [],
            value_fill: float= 0.0,
            ) -> Dict[str, torch.Tensor]:
        if type(views) == list:
            raise Exception("Please feed forward function with dictionary data {view_name_str: torch.Tensor} instead of list")

        model_views = self.view_names 
        if len(testing_views) == 0:
            testing_views = model_views #model defined

        zs_views = {}
        for v_name in model_views:
            if v_name in testing_views and v_name in views:
                data_forward = views[v_name]
            else: #fill when view not in testing forward or view is missing
                data_forward = torch.ones_like(views[v_name])*value_fill #asumming data is available

            zs_views[v_name] = self.views_encoder[v_name](data_forward)
        return {"views:rep": zs_views}

    def forward(self,
            views: Dict[str, torch.Tensor],
            intermediate:bool = True,
            out_norm:bool=False,
            forward_representations_only: bool=False,
            testing_views: list = [],
            value_fill: float= 0.0,
            ) -> Dict[str, torch.Tensor]:
        if type(views) == list:
            raise Exception("Please feed forward function with dictionary data {view_name_str: torch.Tensor} instead of list")
        out_zs_views = self.forward_encoders(views, testing_views=testing_views, value_fill=value_fill )

        out_z_e = self.fusion_module(list(out_zs_views["views:rep"].values())) #carefully, always forward a list
        if forward_representations_only:
            return dict(**out_z_e, **out_zs_views)

        out_y = self.predictive_model(out_z_e["joint_rep"])
        if out_norm:
            out_y_loss = self.apply_softmax(out_y)
        else:
            out_y_loss = out_y

        return_dic = {"prediction": out_y_loss}
        if intermediate:
            return_dic["last_layer"] = out_y
            return dict( **return_dic, **out_zs_views, **out_z_e)
        else:
            return return_dic

    def prepare_batch(self, batch: dict) -> list:
        views_data, views_target = batch["views"], batch["target"]

        if type(views_data) == list:
            print("views as list")
            if "view_names" in batch:
                if len(batch["view_names"]) != 0:
                    views_to_match = batch["view_names"]
            else:
                views_to_match = self.view_names #assuming a order list with view names based on the construction of the class
            views_dict = {views_to_match[i]: value for i, value in enumerate(views_data) }
        elif type(views_data) == dict:
            views_dict = views_data
        else:
            raise Exception("views in batch should be a List or Dict")

        if type(self.criteria) == torch.nn.CrossEntropyLoss:
            views_target = torch.squeeze(views_target)
        return views_dict, views_target

    def corr_criteria(self, zs_views: dict):
        #calculate correlation on representation
        additional_metric = {}
        check_dimensions = [v_value.shape[-1] for v_value in zs_views.values()]
        if len(zs_views) >1 and len(np.unique(check_dimensions)) == 1: #all from same dimensionality and more than 1 view
            zs_views_a = tuple([ (view - view.mean(dim=0))/view.std(dim=0,unbiased=True) for view in zs_views.values()]) # Subtract the mean and normalize on each view
            for i, v_name_i in enumerate(zs_views):
                for j, v_name_j in enumerate(zs_views):
                    if i>j:
                        cov_matrix = zs_views_a[i].T @ zs_views_a[j] / zs_views_a[i].shape[0]
                        additional_metric[f"corr-{v_name_i}-{v_name_j}"] = cov_matrix.diag().abs().mean()
        return additional_metric

    def loss_batch(self, batch: dict) -> dict:
        #calculate values of loss that will not return the model in a forward/transform
        views_dict, views_target = self.prepare_batch(batch)
        out_dic = self(views_dict)
        additional_metric = {}
        if "views:rep" in out_dic:
            additional_metric = self.corr_criteria(out_dic["views:rep"])
        return {"objective": self.criteria(out_dic["prediction"], views_target), **additional_metric}

    #forward for test set
    def transform(self,
            loader: torch.utils.data.DataLoader,
            output=False,
            intermediate=True,
            out_norm=False,
            device:str="",
            args_forward:dict = {},
            **kwargs
            ) -> dict:
        """
        :param loader: a dataloader that matches the structure of that used for training
        :return: transformed views

        #return numpy arrays based on dictionary
        """
        if device == "":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device_used = torch.device(device)

        self.eval() #set batchnorm and dropout off
        self.to(device_used)
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                views_dict, views_target = self.prepare_batch(batch)
                for view_name in views_dict:
                    views_dict[view_name] = views_dict[view_name].to(device_used)
                if output:
                    outputs_ = self(views_dict, intermediate=intermediate, out_norm=out_norm, **args_forward)
                else:
                    outputs_ = self(views_dict, forward_representations_only=True, intermediate=intermediate, **args_forward) #just view encoders
                outputs_ = detach_all(outputs_)

                if batch_idx == 0:
                    outputs = object_to_list(outputs_) #to start append values
                else:
                    collate_all_list(outputs, outputs_) #add to list in cpu
        self.train()
        outputs = stack_all(outputs) #stack with numpy in cpu
        return outputs

    def predict(self, loader: torch.utils.data.DataLoader, out_norm:bool =False, device=""):
        return self.transform( loader, output=True, intermediate=False, out_norm=out_norm,device=device)["prediction"]


class MVFusionMultiLoss(MVFusion):
    def __init__(self,
                 view_encoders: Union[List[nn.Module],Dict[str,nn.Module]],
                 fusion_module: nn.Module,
                 predictive_model: nn.Module,
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 sharing: bool = False,
                 ):
        loss_args["function"] = "custom-multiloss"
        super(MVFusionMultiLoss, self).__init__(view_encoders, fusion_module, predictive_model,
            loss_args=loss_args, view_names=view_names)
        loss_args.pop("function")

        self.aux_predictor_base = copy.deepcopy(self.predictive_model)

        if loss_args.get("sharing"):
            self.sharing = loss_args.pop("sharing")
        else:
            self.sharing= False
        self.aux_predictor = {}
        for v_name in self.view_names:
            if self.sharing:
                self.aux_predictor[v_name] = self.aux_predictor_base
            else:
                self.aux_predictor[v_name] = copy.deepcopy(self.aux_predictor_base)
                self.aux_predictor[v_name].load_state_dict(self.aux_predictor_base.state_dict())
            if not self.fusion_module.get_info_dims().get("feature_pool"): #if not pooling, then change first layer of each predictor
                out_encoder_v_name = self.views_encoder[v_name].get_output_size()
                self.aux_predictor[v_name].update_first_layer(input_features = out_encoder_v_name)
        self.aux_predictor = nn.ModuleDict(self.aux_predictor)

        if "weights_loss" not in loss_args:
            weights_loss = 1
        else:
            weights_loss = loss_args.pop("weights_loss")

        if type(weights_loss) == list:
            if len(weights_loss) == 0:
                self.weights_loss = {v_name: 1 for v_name in self.view_names}
            elif len(weights_loss) == len(self.view_names):
                self.weights_loss = {v_name: weights_loss[i] for i, v_name in enumerate(self.view_names)} #assuming orderer list
        elif type(weights_loss) == dict:
            self.weights_loss = weights_loss
        else: #int-- same value for all
            self.weights_loss = {v_name: weights_loss for v_name in self.view_names}

        self.criteria = loss_args["function"] if "function" in loss_args else get_loss_by_name(**loss_args)
        loss_args["weights_loss"] =weights_loss

    def forward(self, views: Dict[str, torch.Tensor], intermediate = True, out_norm=False):
        if type(views) == list:
            raise Exception("Please feed forward function with dictionary data {view_name_str: torch.Tensor} instead of list")
        out_dic = super(MVFusionMultiLoss, self).forward(views, intermediate = True, out_norm=out_norm)

        out_y_zs = {}
        for v_name in self.view_names:
            out_y_zs[v_name] = self.aux_predictor[v_name]( out_dic["views:rep"][v_name])
        out_dic["views:prediction"] = out_y_zs

        if out_norm:
            for v in out_dic: 
                if ":prediction" in v:
                    for key, value in out_dic[v].items():
                         out_dic[v][key] = self.apply_softmax(value)
        return out_dic

    def loss_batch(self, batch: dict):
        views_dict, views_target = self.prepare_batch(batch)
        out_dic = self(views_dict)
        y_x = out_dic["prediction"]
        yi_xi = out_dic["views:prediction"]
        loss_main = self.criteria(y_x, views_target)

        loss_dic = { }
        loss_aux = 0
        for v_name in self.view_names: 
            loss_dic["loss"+v_name] = self.weights_loss[v_name]*self.criteria(yi_xi[v_name], views_target)
            loss_aux += loss_dic["loss"+v_name]

        if "views:rep" in out_dic:
            additional_metric = self.corr_criteria(out_dic["views:rep"])
            loss_dic = dict(loss_dic, **additional_metric) #concat two dictionaries

        return {"objective": loss_main+loss_aux/len(self.view_names),
                "lossmain":loss_main,"lossaux":loss_aux, **loss_dic}

class HybridFusion(MVFusionMultiLoss): #feature+decision
    def __init__(self,
                 view_encoders: Union[List[nn.Module],Dict[str,nn.Module]],
                 fusion_module_feat: nn.Module,
                 predictive_model: nn.Module,
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 sharing: bool = False,
                 fusion_module_deci: nn.Module = None,
                 ):
        super(HybridFusion, self).__init__(view_encoders, fusion_module_feat, predictive_model,
            loss_args=loss_args, view_names=view_names)
        if fusion_module_deci is not None:
            self.fusion_module_deci = fusion_module_deci
        else:
            self.fusion_module_deci = lambda x: {"joint_rep": torch.mean( torch.stack(x, 1), axis=1 )}

    def forward(self, views: Dict[str, torch.Tensor], intermediate = True, out_norm=False):
        if type(views) == list:
            raise Exception("Please feed forward function with dictionary data {view_name_str: torch.Tensor} instead of list")
        out_dic = super(HybridFusion, self).forward(views, intermediate = True, out_norm=out_norm)
        out_dic["fusion:prediction"] = {"feat": out_dic.pop("prediction") }

        out_y_zs = {}
        for v_name in self.view_names:
            out_y_zs[v_name] = self.aux_predictor[v_name]( out_dic["views:rep"][v_name])
        out_dic["views:prediction"] = out_y_zs
        aux_out = self.fusion_module_deci(list(out_y_zs.values())) 
        out_dic["fusion:prediction"]["dec"] = aux_out.pop("joint_rep")
        if intermediate:
            if "att_views" in aux_out:
                aux_out["att_views_add:dec"] = aux_out.pop("att_views")
            out_dic = dict(**out_dic, **aux_out)

        out_dic["prediction"] = (out_dic["fusion:prediction"]["feat"]+out_dic["fusion:prediction"]["dec"])/2.

        if out_norm:
            for v in out_dic: #
                if ":prediction" in v:
                    for key, value in out_dic[v].items():
                         out_dic[v][key] = self.apply_softmax(value)
                elif "prediction" in v:
                    out_dic[v] = self.apply_softmax(out_dic[v])
        return out_dic

class SVPool(MVFusion):
    #train single-view learning models in a pool, indeepentely between each other
    def __init__(self,
                 view_encoders: Union[List[nn.Module],Dict[str,nn.Module]],
                 predictive_model: nn.Module,
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 ):
        super(SVPool, self).__init__(view_encoders, nn.Identity(), nn.Identity(),
            loss_args=loss_args, view_names=view_names)

        self.view_predictive_models = {}
        for v_name in self.view_names:
            self.view_predictive_models[v_name] = copy.deepcopy(predictive_model)
            self.view_predictive_models[v_name].load_state_dict(predictive_model.state_dict())
        self.view_predictive_models = nn.ModuleDict(self.view_predictive_models)

        self.weights_loss = {v_name: 1 for v_name in self.view_names}

    def forward(self, views: Dict[str, torch.Tensor], intermediate=True, out_norm=False, forward_representations_only: bool=False):
        if type(views) == list:
            raise Exception("Please feed forward function with dictionary data {view_name_str: torch.Tensor} instead of list")
        out_zs_views = self.forward_encoders(views)
        if forward_representations_only:
            return out_zs_views

        out_y_zs = {}
        for v_name in out_zs_views["views:rep"]:
            out_y = self.view_predictive_models[v_name]( out_zs_views["views:rep"][v_name])
            if out_norm:
                out_y_zs[v_name] = self.apply_softmax(out_y)
            else:
                out_y_zs[v_name] = out_y
        return dict(out_zs_views, **{"views:prediction": out_y_zs})

    def loss_batch(self, batch: dict):
        views_dict, views_target = self.prepare_batch(batch)
        out_dic = self(views_dict)
        yi_xi = out_dic["views:prediction"]
        loss_dic = { }
        loss_aux = 0
        for v_name in self.view_names: #for missing needs to be changed
            loss_dic["loss"+v_name] = self.weights_loss[v_name]*self.criteria(yi_xi[v_name], views_target)
            loss_aux += loss_dic["loss"+v_name]

        if "views:rep" in out_dic:
            additional_metric = self.corr_criteria(out_dic["views:rep"])
            loss_dic = dict(loss_dic, **additional_metric) #concat two dictionaries
        return {"objective": loss_aux, **loss_dic}

    def get_sv_models(self):
        return self.view_predictive_models
