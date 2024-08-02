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
            # print(name)
            trainable_params += param.numel()
    for name, param in model.LLM.named_parameters():
        if "classifier" in name:
            # print(name)
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

