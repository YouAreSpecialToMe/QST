import argparse
import os
import pickle
import time

# import GPUtil
from datasets import load_dataset
from evaluate import load
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, TrainingArguments, BitsAndBytesConfig, \
    Trainer, AutoConfig, DataCollatorWithPadding,AutoModelForSequenceClassification
from QSTConfig import QSTConfig
from typing import Dict
from modeling_llama_qst import QSTLlamaForSequenceClassification, LlamaForSequenceClassification

import warnings

# Filter out the specific warning
warnings.filterwarnings("ignore",
                        message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.")

torch.backends.cuda.matmul.allow_tf32 = True


# class MemoryLoggingCallback(TrainerCallback):
#     def __init__(self):
#         super().__init__()
#         self.memory_allocated = []
#
#     def on_step_end(self, args, state, control, **kwargs):
#         initial_memory = GPUtil.getGPUs()[0].memoryUsed
#         self.memory_allocated.append(initial_memory)

def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        # output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        # output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        # output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        if "blackbone" in name:
            param.requires_grad = False
        if "model.layer" in name:
            param.requires_grad = False
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}

GLUE_TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"]
DEFAULT_PAD_TOKEN = "[PAD]"


def train(task, parameters):
    batch_size = parameters[task]["batch_size"]
    model_checkpoint = parameters["model_checkpoint"]
    epoch = parameters[task]["epoch"]
    r = parameters[task]["r"]
    alpha_r = parameters[task]["alpha_r"]
    learning_rate = parameters[task]["learning_rate"]
    max_len = parameters[task]["max_seqlen"]
    qst_checkpoint = parameters['qst_checkpoint']

    actual_task = "mnli" if task == "mnli-mm" else task

    print(f"Loading dataset for task: {actual_task}")
    dataset = load_dataset("glue", task)
    metric = load('glue', task)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, max_length=max_len)

    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    LLM = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, load_in_4bit=True,
                                                         quantization_config=quant_config, torch_dtype=torch.bfloat16,
                                                         num_labels=num_labels,device_map="auto")

    if tokenizer._pad_token is None:
        # smart_tokenizer_and_embedding_resize(
        #     special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        #     tokenizer=tokenizer,
        #     model=LLM,
        # )
        tokenizer.pad_token = tokenizer.eos_token

    sentence1_key, sentence2_key = task_to_keys[task]

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True, padding=True, )
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding=True, )

    encoded_dataset = dataset.map(preprocess_function, batched=True)



    # config = AutoConfig.from_pretrained(model_checkpoint)
    # config.hidden_size = 64



    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
    num_samples = len(encoded_dataset[validation_key])
    num_batches = num_samples // batch_size
    valid_samples = num_batches * batch_size

    encoded_dataset[validation_key] = encoded_dataset[validation_key].select(range(valid_samples))

    config = AutoConfig.from_pretrained(model_checkpoint)
    config.pad_token_id = config.eos_token_id

    LLM.config.torch_dtype = torch.float32

    qst_config = QSTConfig(
        add_layer_norm_before_adapter=False,
        add_layer_norm_after_adapter=True,
        r=r,
        alpha_r=alpha_r,
        dropout=0.1,
        activation="swish",
        fan_in_fan_out=False,
        peft_hidden_size=16  # here
    )

    model = QSTLlamaForSequenceClassification(LLM, config, qst_config)
    model.config.pad_token_id = tokenizer.pad_token_id
    # LLaMA tokenizer may not have correct special tokens set.
    # Check and add them if missing to prevent them from being parsed into different tokens.
    # Note that these are present in the vocabulary.
    # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.

    if qst_checkpoint:
        print("Loading QST from checkpoint.")
        model.load_qst_state(qst_checkpoint)
    else:
        print(f'initing QST modules...')

    # use 16bit as the compute data type, comment it if you want to use 32bit
    for name, module in model.named_modules():
        if 'qst' or 'z' or 'downsample' or 'upsample' in name:
            module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    train_args = TrainingArguments(
        f"{model_checkpoint}-QST-{task}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoch,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        logging_dir=f"{model_checkpoint}-QST-{task}-log",
        logging_strategy="epoch",
        bf16=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    start_time = time.time()
    # memory_callback = MemoryLoggingCallback()

    trainer = Trainer(
        model,
        train_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        # callbacks=[memory_callback]
    )

    trainer.train()
    end_time = time.time()
    results = trainer.evaluate()

    return results, trainer.state.log_history, (end_time - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paramters of QST.")
    parser.add_argument("--batch_size", type=int, required=True, help="batch size")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="model checkpoint")
    parser.add_argument("--qst_checkpoint", type=str, default=None, help="model checkpoint")

    args = parser.parse_args()
    parameters = {
        "model_checkpoint": args.model_checkpoint,
        "qst_checkpoint": args.qst_checkpoint,
        "mnli": {"batch_size": args.batch_size, "epoch": 7, "r": 16, "alpha_r": 16, "max_seqlen": 512,
                 "learning_rate": 5E-04},
        "sst2": {"batch_size": args.batch_size, "epoch": 7, "r": 16, "alpha_r": 16, "max_seqlen": 512,
                 "learning_rate": 5E-04},
        "mrpc": {"batch_size": args.batch_size, "epoch": 7, "r": 16, "alpha_r": 16, "max_seqlen": 512,
                 "learning_rate": 4E-04},
        "cola": {"batch_size": args.batch_size, "epoch": 7, "r": 16, "alpha_r": 16, "max_seqlen": 512,
                 "learning_rate": 5E-04},
        "qnli": {"batch_size": args.batch_size, "epoch": 7, "r": 16, "alpha_r": 16, "max_seqlen": 512,
                 "learning_rate": 4E-04},
        "qqp": {"batch_size": args.batch_size, "epoch": 7, "r": 16, "alpha_r": 16, "max_seqlen": 512,
                "learning_rate": 5E-04},
        "rte": {"batch_size": args.batch_size, "epoch": 7, "r": 16, "alpha_r": 16, "max_seqlen": 512,
                "learning_rate": 5E-04},
        "stsb": {"batch_size": args.batch_size, "epoch": 7, "r": 16, "alpha_r": 16, "max_seqlen": 512,
                 "learning_rate": 4E-04},
    }

    result_dict = {}
    for task in GLUE_TASKS:
        if task == "qnli":
            continue

        result_dict[task] = {}
        result, log, train_time = train(task, parameters)

        values = []
        for elem in log:
            if "eval_loss" not in elem.keys():
                continue
            if task == "cola":
                values.append(elem['eval_matthews_correlation'])
            elif task == "stsb":
                values.append(elem['eval_pearson'])
            else:
                values.append(elem['eval_accuracy'])

        best_acc = max(values)
        result_dict[task]["acc"] = best_acc
        result_dict[task]["time"] = train_time
        # result_dict[task]["memory_usage"] = memory_usage

        print(f"Task:{task}: Best acc {best_acc}, Total training time {train_time}")

    model_name = os.path.basename(parameters["model_checkpoint"])
    with open(f"glue_qst_{task}_{model_name}_{args.batch_size}.pickle", 'wb') as f:
        pickle.dump(result_dict, f)
