import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, PretrainedConfig, \
    AutoModelForSequenceClassification
import math
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
from transformers.activations import get_activation
import bitsandbytes as bnb
from dataclasses import dataclass, field
from transformers.modeling_outputs import SequenceClassifierOutputWithPast, CausalLMOutputWithPast, \
    BaseModelOutputWithPast


@dataclass
class QSTConfig(PretrainedConfig):
    add_layer_norm_before_adapter: bool = False
    add_layer_norm_after_adapter: bool = True
    activation: str = ""
    r: int = 0
    alpha_r: int = 0
    dropout: float = 0.0
    activation: str = ""
    fan_in_fan_out: bool = False,
    peft_hidden_size: int = 64,


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    classifer_params = 0
    for name, param in model.named_parameters():
        # print(name)
        # print(f'Layer: {_} | Device: {param.device}')
        if "score" in name:
            # print(name)
            classifer_params += param.numel()
            param.requires_grad = False
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    for name, param in model.LLM.named_parameters():
        if "classifier" in name:
            classifer_params += param.numel()
            param.requires_grad = True
        all_param += param.numel()
        if param.requires_grad:
            # print(name)
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    print(f"classifer:{classifer_params}")


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


class Activations(nn.Module):
    def __init__(self, activation_type):
        super().__init__()
        self.f = get_activation(activation_type)

    def forward(self, x):
        return self.f(x)


class AdapterLinear(nn.Module):
    #     # Adapter
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 r: int,
                 alpha_r: int,
                 activation=None,
                 add_layer_norm_before_adapter=False,
                 add_layer_norm_after_adapter=False,
                 dropout=0.0,
                 bias=False,
                 ):

        super(AdapterLinear, self).__init__()
        self.adapter_A = nn.Linear(in_features, r,bias=bias)
        nn.init.kaiming_uniform_(self.adapter_A.weight, a=math.sqrt(5))
        self.adapter_B = nn.Linear(r, out_features,bias=bias)
        nn.init.kaiming_uniform_(self.adapter_B.weight, a=math.sqrt(5))
        # nn.init.zeros_(self.adapter_B.weight)
        if activation is not None:
            self.activation = Activations(activation.lower())
        else:
            self.activation = None
        self.add_layer_norm_before_adapter = add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = add_layer_norm_after_adapter

        if self.add_layer_norm_before_adapter:
            self.pre_layer_norm = nn.LayerNorm(in_features)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(out_features)

        if dropout > 0.0:
            self.dropout_layer = nn.Identity()
        else:
            self.dropout_layer = nn.Dropout(p=dropout)

        self.scaling = r / alpha_r

    def set_bias(self, enabled=False):
        self.adapter_A.bias.requires_grad = enabled
        self.adapter_B.bias.requires_grad = enabled

    def forward(self, x):

        x = self.dropout_layer(x)

        if self.add_layer_norm_before_adapter:
            x = self.pre_layer_norm(x)
        x = self.adapter_A(x)
        if self.activation is not None:
            x = self.activation(x)
        y = self.adapter_B(x)
        if self.add_layer_norm_after_adapter:
            y = self.post_layer_norm(y)
        return y
