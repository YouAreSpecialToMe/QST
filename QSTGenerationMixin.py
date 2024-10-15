from transformers import PreTrainedModel, GenerationMixin, LogitsProcessorList, StoppingCriteriaList, GenerationConfig
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import torch


class QSTGenerationMixin(GenerationMixin):
    """
    Custom generation mixin to handle qst_past_key_values, qst_hidden_states, and qst_attentions during generation.
    """

    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            synced_gpus: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            streamer: Optional["BaseStreamer"] = None,
            negative_prompt_ids: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            output_qst_attentions: Optional[bool] = None,
            output_qst_hidden_states: Optional[bool] = None,
            **kwargs
    ):
        # Prepare model_kwargs to include all additional parameters that need to be passed to super().generate()
        model_kwargs = {
            'inputs': inputs,
            'generation_config': generation_config,
            'logits_processor': logits_processor,
            'stopping_criteria': stopping_criteria,
            'prefix_allowed_tokens_fn': prefix_allowed_tokens_fn,
            'synced_gpus': synced_gpus,
            'assistant_model': assistant_model,
            'streamer': streamer,
            'negative_prompt_ids': negative_prompt_ids,
            'negative_prompt_attention_mask': negative_prompt_attention_mask,
            'output_qst_attentions': output_qst_attentions,
            'output_qst_hidden_states': output_qst_hidden_states
        }

        # Incorporate any other keyword arguments
        model_kwargs.update(kwargs)

        # Call the super class generate method using unpacked keyword arguments
        return super().generate(**model_kwargs)

    def _update_model_kwargs_for_generation(
                self,
                outputs,
                model_kwargs,
                is_encoder_decoder=False,
                **kwargs
        ):
            """
            Update model_kwargs with qst_past_key_values for the next generation step.
            """
            # Update qst_past_key_values
            if outputs.qst_past_key_values is not None:
                model_kwargs["qst_past_key_values"] = outputs.qst_past_key_values

            # Call the parent method to handle other updates
            model_kwargs = super()._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=is_encoder_decoder,
                **kwargs
            )
            return model_kwargs
