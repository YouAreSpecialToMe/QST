import os
import pickle
import time

import GPUtil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from datasets import load_dataset, load_metric
import numpy as np
import torch
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    EarlyStoppingCallback, BitsAndBytesConfig, TrainerCallback
from peft import get_peft_model, LoraConfig
from peft.tuners.lora import LoraLayer

class MemoryLoggingCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        # self.initial_memory_allocated = initial_memory_allocated
        self.memory_allocated = []
        # self.memory_cached = []

    def on_step_end(self, args, state, control, **kwargs):
        # allocated = torch.cuda.memory_allocated()
        initial_memory = GPUtil.getGPUs()[0].memoryUsed
        # print(initial_memory)
        # cached = torch.cuda.memory_cached()
        self.memory_allocated.append(initial_memory)
        # self.memory_cached.append(cached)
        # print(
        #     f"Step {state.global_step}, Memory Allocated: {initial_memory}MB")

def find_all_linear_names(bits, model):
    '''
        如果bits是4，使用bitsandbytes库中的bnb.nn.Linear4bit层；
        如果bits是8，使用bitsandbytes库中的bnb.nn.Linear8bitLt层；
        否则，使用torch.nn.Linear层；
        并记录下这些层的名称，保存在lora_module_names集合中。
    '''
    cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            # 只保留最后的名称，前缀不保留
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # 语言模型的输出头，需要16bit精度
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        # print(name)
        # print(f'Layer: {_} | Device: {param.device}')
        if "classifier" in name:
            param.requires_grad = False
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            # print(name)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    # print(model)
    # with open('output.txt', 'w') as f:
    #     print(model, file=f)
    # exit(0)


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    # "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    # "wnli": ("sentence1", "sentence2"),
}

GLUE_TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"]


def train(task, parameters):
    batch_size = parameters[task]["batch_size"]
    model_checkpoint = parameters["model_checkpoint"]
    epoch = parameters[task]["epoch"]
    r = parameters[task]["r"]
    alpha = parameters[task]["alpha"]
    target_module = parameters["target_module"]
    learning_rate = parameters[task]["learning_rate"]
    max_len = parameters[task]["max_seqlen"]

    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    metric = load_metric('glue', actual_task)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, max_length=max_len)
    sentence1_key, sentence2_key = task_to_keys[task]

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    # print(encoded_dataset)
    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, quantization_config=quant_config,torch_dtype=torch.float32,
                                                               num_labels=num_labels)
    model.config.torch_dtype = torch.float32
    target_module = find_all_linear_names(4, model)

    config = LoraConfig(
        peft_type="LORA", task_type="SEQ_CLS", r=r, lora_alpha=alpha, target_modules=target_module,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    # exit(0)
    # print(model)
    # exit(0)

    # bf16 = True
    for name, module in model.named_modules():
        # 设置一些层的数据类型
        # if isinstance(module, LoraLayer):
        #     # 如果模型是LoraLayer
        #     if bf16:
        #         # 且使用args.bf16格式，但是只有某些GPU支持
        #         module = module.to(torch.bfloat16)
        if 'norm' in name:
            # 如果是归一化层，transformers中有很多layernorm层
            module = module.to(torch.float32)
        # if 'lm_head' in name or 'embed_tokens' in name:
        #     # 如果lm_head或embed_tokens等，也就是输出层和输入层
        #     if hasattr(module, 'weight'):
        #         # 而且是权重（非bias参数）
        #         if bf16 and module.weight.dtype == torch.float32:
        #             # 如果使用bf16格式，且模型权重的数据类型为torch.float32，
        #             # 则将该层设置为bfloat16数据格式
        #             module = module.to(torch.bfloat16)


    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    args = TrainingArguments(
        output_dir=f"{model_checkpoint}-lora-finetuned-{task}",
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
        fp16=True
    )

    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
    if len(encoded_dataset["train"]) > 20000:
        encoded_dataset["train"] = encoded_dataset["train"].select(range(20000))
    if len(encoded_dataset[validation_key]) > 3000:
        encoded_dataset[validation_key] = encoded_dataset[validation_key].select(range(3000))


    start_time = time.time()
    memory_callback = MemoryLoggingCallback()
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[memory_callback]
    )

    trainer.train()
    end_time = time.time()

    results = trainer.evaluate()
    # print(results)

    return results, trainer.state.log_history, (end_time - start_time), max(memory_callback.memory_allocated)


if __name__ == "__main__":
    parameters = {
        "model_checkpoint": "/home/zzx/pythonproject/LLM/huggingface-demos/experiments/faster_generation/opt-1.3b",
        # "model_checkpoint": "opt-2.7b",
        "target_module": ["query", "value"],
        "mnli": {"batch_size": 4, "epoch": 7, "r": 16, "alpha": 16, "max_seqlen": 128, "learning_rate": 5E-04},
        "sst2": {"batch_size": 4, "epoch": 7, "r": 16, "alpha": 16, "max_seqlen": 128, "learning_rate": 5E-04},
        "mrpc": {"batch_size": 4, "epoch": 7, "r": 16, "alpha": 16, "max_seqlen": 128, "learning_rate": 4E-04},
        "cola": {"batch_size": 16, "epoch": 7, "r": 16, "alpha": 16, "max_seqlen": 128, "learning_rate": 4E-04},
        "qnli": {"batch_size": 4, "epoch": 7, "r": 16, "alpha": 16, "max_seqlen": 128, "learning_rate": 4E-04},
        "qqp": {"batch_size": 4, "epoch": 7, "r": 16, "alpha": 16, "max_seqlen": 128, "learning_rate": 5E-04},
        "rte": {"batch_size": 4, "epoch": 7, "r": 16, "alpha": 16, "max_seqlen": 128, "learning_rate": 5E-04},
        "stsb": {"batch_size": 4, "epoch": 7, "r": 16, "alpha": 16, "max_seqlen": 128, "learning_rate": 4E-04},
    }

    # parameters_robert_l = {
    #     "model_checkpoint": "roberta-large",
    #     "target_module": ["query", "value"],
    #     "mnli": {"batch_size": 16, "epoch": 10, "r": 8, "alpha": 16, "max_seqlen": 128, "learning_rate": 3E-04},
    #     "sst2": {"batch_size": 64, "epoch": 10, "r": 8, "alpha": 16, "max_seqlen": 128, "learning_rate": 4E-04},
    #     "mrpc": {"batch_size": 16, "epoch": 20, "r": 8, "alpha": 16, "max_seqlen": 128, "learning_rate": 3E-04},
    #     "cola": {"batch_size": 32, "epoch": 20, "r": 8, "alpha": 16, "max_seqlen": 128, "learning_rate": 2E-04},
    #     "qnli": {"batch_size": 4, "epoch": 10, "r": 8, "alpha": 16, "max_seqlen": 128, "learning_rate": 2E-04},
    #     "qqp": {"batch_size": 4, "epoch": 20, "r": 8, "alpha": 16, "max_seqlen": 128, "learning_rate": 3E-04},
    #     "rte": {"batch_size": 8, "epoch": 20, "r": 8, "alpha": 16, "max_seqlen": 128, "learning_rate": 4E-04},
    #     "stsb": {"batch_size": 8, "epoch": 30, "r": 8, "alpha": 16, "max_seqlen": 128, "learning_rate": 2E-04},
    # }

    result_dict = {}
    for task in GLUE_TASKS:
        # task = "stsb"
        if task == "qnli":
            continue

        result_dict[task] = {}
        result, log, train_time, memory_usage = train(task, parameters)
        # result_dict[task]["result"] = result
        # result_dict["log"] = log

        values = []
        for elem in log:
            print(f"elem: {elem}")
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
        result_dict[task]["memory_usage"] = memory_usage

        print(f"Task:{task}: Best acc {best_acc}, Total training time {train_time}, Memory usage {memory_usage}")

    model_name = os.path.basename(parameters["model_checkpoint"])
    with open(f"glue_qlora_{task}_{model_name}.pickle", 'wb') as f:
        pickle.dump(result_dict, f)

# task = "cola"
# model_checkpoint = "roberta-base"
# batch_size = 32
#
# actual_task = "mnli" if task == "mnli-mm" else task
# dataset = load_dataset("glue", actual_task)
# metric = load_metric('glue', actual_task)
#
# # test metrics
# # fake_preds = np.random.randint(0, 2, size=(64,))
# # fake_labels = np.random.randint(0, 2, size=(64,))
# # print(metric.compute(predictions=fake_preds, references=fake_labels))
#
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)


# sentence1_key, sentence2_key = task_to_keys[task]
# if sentence2_key is None:
#     print(f"Sentence: {dataset['train'][0][sentence1_key]}")
# else:
#     print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
#     print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")

# def preprocess_function(examples):
#     if sentence2_key is None:
#         return tokenizer(examples[sentence1_key], truncation=True)
#     return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

# encoded_dataset = dataset.map(preprocess_function, batched=True)
# print(encoded_dataset)
# num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
# model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
# # print(model)
#
# moeconfig = MoeLoraConfig(
#     peft_type="LORA", task_type="CAUSAL_LM", r=8, lora_alpha=32, target_modules=["query","value"],
#     lora_dropout=0.01, num_experts=8
# )
#
# config = LoraConfig(
#     peft_type="LORA", task_type="CAUSAL_LM", r=8, lora_alpha=8, target_modules=["query","value"],
#     lora_dropout=0.01,
# )

# def print_trainable_parameters(model):
#     """
#     Prints the number of trainable parameters in the model.
#     """
#     trainable_params = 0
#     all_param = 0
#     for name, param in model.named_parameters():
#         # print(f'Layer: {_} | Device: {param.device}')
#         if "classifier" in name:
#             param.requires_grad = True
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#     print(
#         f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
#     )
#     # print(model)
#     with open('output.txt', 'w') as f:
#         print(model, file=f)
#
# # print_trainable_parameters(model)
# model = get_peft_model(model, config)
# print_trainable_parameters(model)
#
# metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
# # model_name = model_checkpoint.split("/")[-1]
#
# args = TrainingArguments(
#     f"{model_checkpoint}-finetuned-{task}",
#     evaluation_strategy = "epoch",
#     save_strategy = "epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     num_train_epochs=80,
#     weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model=metric_name,
# )
#
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     if task != "stsb":
#         predictions = np.argmax(predictions, axis=1)
#     else:
#         predictions = predictions[:, 0]
#     return metric.compute(predictions=predictions, references=labels)
#
# validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
# trainer = Trainer(
#     model,
#     args,
#     train_dataset=encoded_dataset["train"],
#     eval_dataset=encoded_dataset[validation_key],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )
#
# trainer.train()
# print(trainer.evaluate())
