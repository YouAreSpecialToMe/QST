from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput, SequenceClassifierOutputWithPast, CausalLMOutputWithPast
)


@dataclass
class LSTBaseModelOutput(BaseModelOutput):
    last_hidden_states: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    last_lst_hidden_state: torch.FloatTensor = None
    lst_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    lst_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class LSTBaseModelOutputWithPast(BaseModelOutputWithPast):
    last_hidden_states: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    last_lst_hidden_states: torch.FloatTensor = None
    lst_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    lst_attentions: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    lst_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class SideBaseModelOutputWithPastAndCrossAttentions(BaseModelOutputWithPastAndCrossAttentions):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    last_side_hidden_state: torch.FloatTensor = None
    side_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    side_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    side_attentions: Optional[Tuple[torch.FloatTensor]] = None
    side_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SideSeq2SeqLMOutput(Seq2SeqLMOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    side_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class SideSequenceClassifierOutputWithPast(SequenceClassifierOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # last_side_hidden_state: torch.FloatTensor = None
    side_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    side_attentions: Optional[Tuple[torch.FloatTensor]] = None
    side_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

@dataclass
class LSTCausalLMOutputWithPast(CausalLMOutputWithPast):
    lst_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    lst_attentions: Optional[Tuple[torch.FloatTensor]] = None
    lst_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

@dataclass
class LSTSequenceClassifierOutputWithPast(SequenceClassifierOutputWithPast):
    lst_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    lst_attentions: Optional[Tuple[torch.FloatTensor]] = None
    lst_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
