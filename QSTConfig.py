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
    # print(model)
    # with open('output.txt', 'w') as f:
    #     print(model, file=f)
    # exit(0)


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
        # print(f"in: {in_features}")
        # print(f"out: {out_features}")
        # exit(0)
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

        # x = x.to(self.adapter_A.weight.device)
        x = self.dropout_layer(x)

        if self.add_layer_norm_before_adapter:
            x = self.pre_layer_norm(x)
        # print(f"x: {x.dtype}")
        # print(f"weight: {self.adapter_A.weight.dtype}")
        # exit(0)
        x = self.adapter_A(x)
        if self.activation is not None:
            x = self.activation(x)
        y = self.adapter_B(x)
        if self.add_layer_norm_after_adapter:
            y = self.post_layer_norm(y)
        return y


class MoEAdaptorsLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 r: int,
                 alpha_r: int,
                 activation,
                 num_expert: int,
                 routing_strategy: str = "gating",
                 weight_average: bool = False,
                 add_layer_norm_before_adapter=False,
                 add_layer_norm_after_adapter=False,
                 dropout=0.0,
                 ):
        super(MoEAdaptorsLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_expert = num_expert
        self.experts = nn.ModuleList([])
        self.routing_strategy = routing_strategy
        self.scaling = r / alpha_r
        for i in range(self.num_expert):
            self.experts.append(AdapterLinear(in_features, out_features, r, activation, add_layer_norm_before_adapter,
                                              add_layer_norm_after_adapter, dropout))

        if self.routing_strategy == "gating":
            self.gate = nn.Linear(self.out_features, self.num_expert, bias=False)
            # self.gate.weight = self.gate.weight.to(torch.bfloat16)

    def forward(self, x):

        len_x = x.size()
        if len(len_x) == 3:
            bsz, seq_len, dim = x.size()
        else:
            bsz, dim = x.size()

        x = x.view(-1, dim)

        # print(f"x2: {x.device}")
        # print(f"x2: {self.gate.weight.device}")
        # exit(0)
        logits_gate = self.gate(x)
        prob_gate = F.softmax(logits_gate, dim=-1)
        gate = torch.argmax(prob_gate, dim=-1)

        order = gate.argsort(0)
        num_tokens = F.one_hot(gate, self.num_expert).gt(0).sum(0)
        gate_load = num_tokens.clone()
        x = x[order]  # reorder according to expert number
        x = x.split(num_tokens.tolist(), dim=0)  # a list of length self.num_experts

        prob_gate = prob_gate.gather(dim=1, index=gate.unsqueeze(1))
        prob_gate = prob_gate[order]
        prob_gate = prob_gate.split(num_tokens.tolist(), dim=0)

        def forward_expert(input_x, prob_x, expert_idx):
            input_x = self.experts[expert_idx].forward(input_x) * self.scaling
            input_x = input_x * prob_x
            return input_x

        x = [forward_expert(x[i], prob_gate[i], i) for i in range(self.num_expert)]
        x = torch.vstack(x)
        x = x[order.argsort(0)]  # restore original order
        if len(len_x) == 3:
            x = x.view(bsz, seq_len, self.out_features)

        return x


class LSTQuant(nn.Module):
    def __init__(self, config, LLM):
        super(LSTQuant, self).__init__()
        self.config = config
        self.device = config.cuda
        self.LLM = LLM
        # self.tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)
        # self.emb_peft_type = config.emb_peft_type.lower()
        self.linear_peft_type = config.linear_peft_type.lower()
        self.r = config.r
        self.alpha_r = config.alpha_r
        self.num_expert = config.num_expert
        self.routing_strategy = config.routing_strategy.lower()
        self.weight_average = config.weight_average
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.dropout = config.dropout
        self.layers = config.num_hidden_layers
        self.activation = config.activation
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.task_type = config.task_type.lower()

        self.fan_in_fan_out = config.fan_in_fan_out
        # self.LLM.score.require_grad = True

        self.gates = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.5])) for _ in range(self.layers)])

        if self.linear_peft_type == "lora":
            self.pefts = nn.ModuleList(
                [MoEAdaptorsLinear(in_features=self.hidden_size, out_features=self.hidden_size, r=self.r,
                                   alpha_r=self.alpha_r, activation=None,
                                   num_expert=self.num_expert,
                                   routing_strategy=self.routing_strategy,
                                   weight_average=self.weight_average,
                                   add_layer_norm_after_adapter=self.add_layer_norm_after_adapter,
                                   add_layer_norm_before_adapter=self.add_layer_norm_before_adapter,
                                   dropout=self.dropout).to(self.device) for _ in range(self.layers)])
        elif self.linear_peft_type == "adaptor":
            self.pefts = nn.ModuleList(
                [MoEAdaptorsLinear(in_features=self.hidden_size, out_features=self.hidden_size, r=self.r,
                                   alpha_r=self.alpha_r, activation=self.activation,
                                   num_expert=self.num_expert,
                                   routing_strategy=self.routing_strategy,
                                   weight_average=self.weight_average,
                                   add_layer_norm_after_adapter=self.add_layer_norm_after_adapter,
                                   add_layer_norm_before_adapter=self.add_layer_norm_before_adapter,
                                   dropout=self.dropout).to(self.device) for _ in range(self.layers)])
        elif self.linear_peft_type == "transformer":
            self.pefts = nn.ModuleList(
                [nn.TransformerDecoderLayer(self.hidden_size, self.config.nhead, self.hidden_size,
                                            self.dropout) for _ in range(self.layers)])

    def _forward_sc(self,
                    input_ids: Optional[torch.LongTensor] = None,
                    attention_mask: Optional[torch.FloatTensor] = None,
                    head_mask: Optional[torch.FloatTensor] = None,
                    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                    inputs_embeds: Optional[torch.FloatTensor] = None,
                    labels: Optional[torch.LongTensor] = None,
                    use_cache: Optional[bool] = None,
                    output_attentions: Optional[bool] = None,
                    output_hidden_states: Optional[bool] = None,
                    return_dict: Optional[bool] = None, ) -> Union[Tuple, SequenceClassifierOutputWithPast]:

        # print(return_dict)
        with torch.no_grad():
            transformer_outputs = self.LLM.model(input_ids=input_ids, attention_mask=attention_mask,
                                                 head_mask=head_mask,
                                                 # labels=labels,
                                                 past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                                                 use_cache=use_cache, output_attentions=output_attentions,
                                                 output_hidden_states=True, return_dict=return_dict, )

        # print(len(transformer_outputs))
        # idx = 0
        # use_cache = use_cache if use_cache is not None else self.LLM.model.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.LLM.model.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.LLM.model.config.use_return_dict

        # print(use_cache)
        # print(output_attentions)
        # if use_cache:
        #     hidden_states = transformer_outputs[2]
        # else:
        #     hidden_states = transformer_outputs[1]

        # print(transformer_outputs[0].shape)
        # print(len(transformer_outputs[1]))
        # print(len(transformer_outputs[2]))

        # print(transformer_outputs.)
        # print(transformer_outputs.past_key_values.shape)
        # print(transformer_outputs.hidden_states.shape)
        # print(transformer_outputs.attentions.shape)
        # exit(0)
        hidden_states = transformer_outputs.hidden_states
        # print(hidden_states.shape)
        # exit(0)

        # print(transformer_outputs.hidden_states[0].shape)

        # new_hidden_states = [hidden_states[0]]
        # with torch.autocast("cuda"):
        h_x = hidden_states[1]
        # print(h_x.shape)
        x = hidden_states[0]
        # print(x.shape)
        for i in range(self.layers):
            z = torch.sigmoid(self.gates[i])

            x = z * x + (1 - z) * h_x

            x = self.pefts[i](x)
            # new_hidden_states.append(x)
            h_x = hidden_states[i + 1]
            # hidden_states[i] = x

        # # with torch.no_grad():
        # print(x.shape)
        # exit(0)
        logits = self.LLM.score(x)
        # print(logits)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.LLM.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.LLM.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1
                # logger.warning(
                #     f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                #     "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                # )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        # print(f"logits:{pooled_logits}")

        loss = None
        if labels is not None:
            if self.LLM.config.problem_type is None:
                if self.num_labels == 1:
                    self.LLM.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.LLM.config.problem_type = "single_label_classification"
                else:
                    self.LLM.config.problem_type = "multi_label_classification"

            if self.LLM.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.LLM.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
                # print(f"loss:{loss}")
            elif self.LLM.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        # print(pooled_logits.shape)
        # exit(0)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            if not output_hidden_states:
                if not output_attentions:
                    output = output[:-1]
                else:
                    output = output[:-2] + (output[-1],)
            return ((loss,) + output) if loss is not None else output

        # print(pooled_logits.shape)
        # print(labels.shape)
        # exit(0)

        return (loss, pooled_logits)
        # return SequenceClassifierOutputWithPast(
        #     loss=loss,
        #     logits=pooled_logits,
        #     past_key_values=transformer_outputs.past_key_values,
        #     hidden_states=None,
        #     attentions=transformer_outputs.attentions,
        # )

    def _forward_clm(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        outputs = self.LLM.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        hidden_states = outputs.hidden_states

        # new_hidden_states = [hidden_states[0]]
        with torch.autocast("cuda"):
            h_x = hidden_states[1]
            x = hidden_states[0]
            for i in range(self.layers):
                z = torch.sigmoid(self.gates[i])

                x = z * x + (1 - z) * h_x

                x = self.pefts[i](x)
                # new_hidden_states.append(x)
                h_x = hidden_states[i + 1]

            # with torch.no_grad():
            logits = self.LLM.lm_head(x).contiguous()

        # logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None, ):
        if self.task_type == "sc":
            return self._forward_sc(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask,
                                    past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                                    use_cache=use_cache, output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states, return_dict=return_dict, labels=labels)
        # elif self.task_type == "clm":

    # def _set_adaptors(self):
    #
    #     self.pefts = nn.ModuleDict([])
    #     self.gates = nn.ParameterDict([])
    #     # exit(0)
    #     # for idx, (name, module) in enumerate(self.LLM.named_modules()):
    #     #
    #     #     for param in module.parameters():
    #     #         param.requires_grad = False
    #     #
    #     #     if isinstance(module, nn.Embedding):
    #     #         in_features, out_features = module.num_embeddings, module.embedding_dim
    #     #         self.gates[str(idx)] = nn.Parameter(torch.tensor([0.5]))
    #     #         # self.gates.append(nn.Linear(out_features * 2, out_features))
    #     #         if self.emb_peft_type == "lora":
    #     #             self.pefts[str(idx)] = MoEAdaptorsLinear(in_features=in_features, out_features=out_features,
    #     #                                                      r=self.r,
    #     #                                                      alpha_r=self.alpha_r, activation=None,
    #     #                                                      num_expert=self.num_expert,
    #     #                                                      routing_strategy=self.routing_strategy,
    #     #                                                      weight_average=self.weight_average,
    #     #                                                      add_layer_norm_after_adapter=self.add_layer_norm_after_adapter,
    #     #                                                      add_layer_norm_before_adapter=self.add_layer_norm_before_adapter,
    #     #                                                      dropout=self.dropout).to(self.device)
    #     #             # self.pefts.append(MoEAdaptorsLinear(in_features=in_features, out_features=out_features, r=self.r,
    #     #             #                                     alpha_r=self.alpha_r, activation=None,
    #     #             #                                     num_expert=self.num_expert,
    #     #             #                                     routing_strategy=self.routing_strategy,
    #     #             #                                     weight_average=self.weight_average,
    #     #             #                                     add_layer_norm_after_adapter=self.add_layer_norm_after_adapter,
    #     #             #                                     add_layer_norm_before_adapter=self.add_layer_norm_before_adapter,
    #     #             #                                     dropout=self.dropout).to(self.device))
    #     #         elif self.emb_peft_type == "adaptor":
    #     #             # adaptors
    #     #             self.pefts[str(idx)] = MoEAdaptorsLinear(in_features=in_features, out_features=out_features,
    #     #                                                      r=self.r,
    #     #                                                      alpha_r=self.alpha_r, activation=self.activation,
    #     #                                                      num_expert=self.num_expert,
    #     #                                                      routing_strategy=self.routing_strategy,
    #     #                                                      weight_average=self.weight_average,
    #     #                                                      add_layer_norm_after_adapter=self.add_layer_norm_after_adapter,
    #     #                                                      add_layer_norm_before_adapter=self.add_layer_norm_before_adapter,
    #     #                                                      dropout=self.dropout)
    #     #             # self.pefts.append(MoEAdaptorsLinear(in_features=in_features, out_features=out_features, r=self.r,
    #     #             #                                     alpha_r=self.alpha_r, activation=self.activation,
    #     #             #                                     num_expert=self.num_expert,
    #     #             #                                     routing_strategy=self.routing_strategy,
    #     #             #                                     weight_average=self.weight_average,
    #     #             #                                     add_layer_norm_after_adapter=self.add_layer_norm_after_adapter,
    #     #             #                                     add_layer_norm_before_adapter=self.add_layer_norm_before_adapter,
    #     #             #                                     dropout=self.dropout))
    #     #         else:
    #     #             raise NotImplementedError
    #     #
    #     #     if isinstance(module, bnb.nn.Linear8bitLt) or isinstance(module, bnb.nn.Linear4bit):
    #     #         in_features, out_features = module.in_features, module.out_features
    #     #         self.gates[str(idx)] = nn.Parameter(torch.tensor([0.5]))
    #     #         # self.gates.append(nn.Linear(out_features * 2, out_features))
    #     #         if self.linear_peft_type == "lora":
    #     #             self.pefts[str(idx)] = MoEAdaptorsLinear(in_features=in_features, out_features=out_features,
    #     #                                                      r=self.r,
    #     #                                                      alpha_r=self.alpha_r, activation=None,
    #     #                                                      num_expert=self.num_expert,
    #     #                                                      routing_strategy=self.routing_strategy,
    #     #                                                      weight_average=self.weight_average,
    #     #                                                      add_layer_norm_after_adapter=self.add_layer_norm_after_adapter,
    #     #                                                      add_layer_norm_before_adapter=self.add_layer_norm_before_adapter,
    #     #                                                      dropout=self.dropout)
    #     #             # self.pefts.append(MoEAdaptorsLinear(in_features=in_features, out_features=out_features, r=self.r,
    #     #             #                                     alpha_r=self.alpha_r, activation=None,
    #     #             #                                     num_expert=self.num_expert,
    #     #             #                                     routing_strategy=self.routing_strategy,
    #     #             #                                     weight_average=self.weight_average,
    #     #             #                                     add_layer_norm_after_adapter=self.add_layer_norm_after_adapter,
    #     #             #                                     add_layer_norm_before_adapter=self.add_layer_norm_before_adapter,
    #     #             #                                     dropout=self.dropout))
    #     #         elif self.linear_peft_type == "adaptor":
    #     #             self.pefts[str(idx)] = MoEAdaptorsLinear(in_features=in_features, out_features=out_features,
    #     #                                                      r=self.r,
    #     #                                                      alpha_r=self.alpha_r, activation=self.activation,
    #     #                                                      num_expert=self.num_expert,
    #     #                                                      routing_strategy=self.routing_strategy,
    #     #                                                      weight_average=self.weight_average,
    #     #                                                      add_layer_norm_after_adapter=self.add_layer_norm_after_adapter,
    #     #                                                      add_layer_norm_before_adapter=self.add_layer_norm_before_adapter,
    #     #                                                      dropout=self.dropout)
    #     #             # self.pefts.append(MoEAdaptorsLinear(in_features=in_features, out_features=out_features, r=self.r,
    #     #             #                                     alpha_r=self.alpha_r, activation=self.activation,
    #     #             #                                     num_expert=self.num_expert,
    #     #             #                                     routing_strategy=self.routing_strategy,
    #     #             #                                     weight_average=self.weight_average,
    #     #             #                                     add_layer_norm_after_adapter=self.add_layer_norm_after_adapter,
    #     #             #                                     add_layer_norm_before_adapter=self.add_layer_norm_before_adapter,
    #     #             #                                     dropout=self.dropout))
    #     #         elif self.linear_peft_type == "transformer":
    #     #             self.pefts[str(idx)] = nn.TransformerDecoderLayer(in_features, self.config.nhead, out_features,
    #     #                                                               self.dropout)
    #     #             # self.pefts.append(
    #     #             #     nn.TransformerDecoderLayer(in_features, self.config.nhead, out_features, self.dropout))
    #     #         else:
    #     #             raise NotImplementedError
    #     #     else:
    #     #         continue
    #     #
    #     # # def forward(self, input_ids,attention_mask=None,labels=None):
    #     # #     h_x1 = input_ids
    #     # #     for idx, (name, module) in enumerate(self.LLM.named_modules()):
    #     # #         with torch.no_grad():
    #     # #             h_x2 = module.forward(h_x1)
    #     # #
    #     # #         if str(idx) not in self.pefts:
    #     # #             h_x1 = h_x2
    #     # #             continue
    #     # #         y = self.pefts[str(idx)](input_ids)
    #     # #
    #     # #         combined = torch.cat([h_x2, y], dim=-1)
    #     # #         z = torch.sigmoid(self.gates[str(int)](combined))
    #     # #         x2 = z * y + (1 - z) * h_x2
    #     # #
    #     # #         h_x1 = h_x2
    #     # #         x = x2
    #     #
    #     # # return x
    #
    #     def _forward_sc(
    #             self,
    #             input_ids: Optional[torch.LongTensor] = None,
    #             attention_mask: Optional[torch.FloatTensor] = None,
    #             head_mask: Optional[torch.FloatTensor] = None,
    #             past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    #             inputs_embeds: Optional[torch.FloatTensor] = None,
    #             labels: Optional[torch.LongTensor] = None,
    #             use_cache: Optional[bool] = None,
    #             output_attentions: Optional[bool] = None,
    #             output_hidden_states: Optional[bool] = None,
    #             return_dict: Optional[bool] = None,
    #     ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
    #
    #         with torch.no_grad():
    #             transformer_outputs = self.LLM(
    #                 input_ids,
    #                 past_key_values=past_key_values,
    #                 attention_mask=attention_mask,
    #                 head_mask=head_mask,
    #                 inputs_embeds=inputs_embeds,
    #                 use_cache=use_cache,
    #                 output_attentions=output_attentions,
    #                 output_hidden_states=output_hidden_states,
    #                 return_dict=return_dict,
    #             )
    #
    #             hidden_states = transformer_outputs[0]
    #
    #         for idx, h_x in enumerate(hidden_states):
    #             if str(idx) not in self.pefts:
    #                 continue
    #
    #
    #
    #         if input_ids is not None:
    #             batch_size, sequence_length = input_ids.shape[:2]
    #         else:
    #             batch_size, sequence_length = inputs_embeds.shape[:2]
    #
    #         if self.LLM.config.pad_token_id is None:
    #             sequence_lengths = -1
    #         else:
    #             if input_ids is not None:
    #                 sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
    #             else:
    #                 sequence_lengths = -1
    #                 logger.warning(
    #                     f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
    #                     "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
    #                 )
    #
    #         loss = None
    #         if labels is not None:
    #             if self.LLM.config.problem_type is None:
    #                 if self.LLM.num_labels == 1:
    #                     self.LLM.config.problem_type = "regression"
    #                 elif self.LLM.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
    #                     self.LLM.config.problem_type = "single_label_classification"
    #                 else:
    #                     self.LLM.config.problem_type = "multi_label_classification"
    #
    #             if self.LLM.config.problem_type == "regression":
    #                 loss_fct = MSELoss()
    #                 if self.LLM.num_labels == 1:
    #                     loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
    #                 else:
    #                     loss = loss_fct(pooled_logits, labels)
    #             elif self.LLM.config.problem_type == "single_label_classification":
    #                 loss_fct = CrossEntropyLoss()
    #                 loss = loss_fct(pooled_logits.view(-1, self.LLM.num_labels), labels.view(-1))
    #             elif self.LLM.config.problem_type == "multi_label_classification":
    #                 loss_fct = BCEWithLogitsLoss()
    #                 loss = loss_fct(pooled_logits, labels)
    #         if not return_dict:
    #             output = (pooled_logits,) + transformer_outputs[1:]
    #             return ((loss,) + output) if loss is not None else output
    #
    #         return SequenceClassifierOutputWithPast(
    #             loss=loss,
    #             logits=pooled_logits,
    #             past_key_values=transformer_outputs.past_key_values,
    #             hidden_states=transformer_outputs.hidden_states,
    #             attentions=transformer_outputs.attentions,
    #         )
