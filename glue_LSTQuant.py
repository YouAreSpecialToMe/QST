import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import load_dataset, load_metric
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, BitsAndBytesConfig, \
    Trainer, AutoConfig, OPTForSequenceClassification, DataCollatorWithPadding
from LSTQuant import LSTQuant, LSTQuantConfig, print_trainable_parameters, AdapterLinear
from modeling_opt_lst import LSTOPTForCausalLM, LSTOPTForSequenceClassification

torch.backends.cuda.matmul.allow_tf32 = True


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
        if "blackbone" in name:
            # print(name)
            # classifer_params += param.numel()
            param.requires_grad = False
        if "score" in name:
            param.requires_grad = True
        all_param += param.numel()
        if param.requires_grad:
            print(name)
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    # print(f"classifer:{classifer_params}")
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
    num_experts = parameters["num_experts"]

    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    metric = load_metric('glue', actual_task)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, max_length=max_len)

    sentence1_key, sentence2_key = task_to_keys[task]

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True, padding=True, )
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding=True, )

    # print(preprocess_function(dataset['train'][:5]))
    # exit(0)
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    # encoded_dataset["train"] = encoded_dataset["train"].remove_columns('attention_mask')

    # print(encoded_dataset["train"][:5])
    # exit(0)
    # print(encoded_dataset)
    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
    # print(num_labels)
    # exit(0)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
    # encoded_dataset[validation_key] = encoded_dataset[validation_key].remove_columns("attention_mask")
    num_samples = len(encoded_dataset[validation_key])
    num_batches = num_samples // batch_size
    valid_samples = num_batches * batch_size

    encoded_dataset[validation_key] = encoded_dataset[validation_key].select(range(valid_samples))

    config = AutoConfig.from_pretrained(model_checkpoint)
    # print(config.layerdrop)
    # exit(0)
    LLM = OPTForSequenceClassification.from_pretrained(model_checkpoint, load_in_4bit=True,
                                                       quantization_config=quant_config, torch_dtype=torch.float32,
                                                       num_labels=num_labels)
    # help(LLM.forward)
    # exit(0)
    if len(encoded_dataset["train"]) > 20000:
        encoded_dataset["train"] = encoded_dataset["train"].select(range(20000))
    if len(encoded_dataset[validation_key]) > 3000:
        encoded_dataset[validation_key] = encoded_dataset[validation_key].select(range(3000))
    # model = LLM
    # print(model)
    LLM.config.torch_dtype = torch.float32
    # exit(0)
    LSTconfig = LSTQuantConfig(
        add_layer_norm_before_adapter=False,
        add_layer_norm_after_adapter=True,
        word_embed_proj_dim=config.word_embed_proj_dim,
        max_position_embeddings=config.max_position_embeddings,
        # _remove_final_layer_norm =config._remove_final_layer_norm,
        # do_layer_norm_before = config.do_layer_norm_before,
        model_checkpoint="opt-1.3b",
        emb_peft_type="adaptor",
        linear_peft_type="adaptor",
        r=16,
        alpha_r=16,
        num_expert=1,
        routing_strategy="gating",
        weight_average=False,
        dropout=0.1,
        activation="swish",
        cuda="cuda:0",
        fan_in_fan_out=False,
        num_labels=num_labels,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        peft_hidden_size=16
    )

    model = LSTOPTForSequenceClassification(LLM, config, LSTconfig)

    for name, module in model.named_modules():
        # 设置一些层的数据类型
        # if isinstance(module, AdapterLinear):
        #     # 如果模型是LoraLayer
        #     module = module.to(torch.float16)
        if 'norm' in name:
            # 如果是归一化层，transformers中有很多layernorm层
            # if not "post_layer_norm" or "post_layer_norm" in name:
            module = module.to(torch.float32)
        # if 'lm_head' in name or 'embed_tokens' in name or 'score' in name or 'upsample' in name:
        #     # 如果lm_head或embed_tokens等，也就是输出层和输入层
        #     if hasattr(module, 'weight'):
        #         # 而且是权重（非bias参数）
        #         if module.weight.dtype == torch.float32:
        #             # 如果使用bf16格式，且模型权重的数据类型为torch.float32，
        #             # 则将该层设置为bfloat16数据格式
        #             module = module.to(torch.float16)
    print(model)
    # exit(0)
    print_trainable_parameters(model)
    # exit(0)

    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # print(len(predictions))
        # print(predictions.shape)
        # for t in predictions:
        #     print(len(t))
        # # print(predictions[1].shape)
        # # print(predictions[2].shape)
        # print(labels.shape)
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    args = TrainingArguments(
        f"{model_checkpoint}-LSTQuant-finetuned-{task}",
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
        logging_dir=f"{model_checkpoint}-LSTQuant-finetuned-{task}-log",
        logging_strategy="epoch",
        fp16=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    results = trainer.evaluate()
    # print(results)

    return results, trainer.state.log_history


if __name__ == "__main__":
    parameters_robert_b = {
        "model_checkpoint": "/home/zzx/pythonproject/LLM/huggingface-demos/experiments/faster_generation/opt-1.3b",
        # "model_checkpoint": "opt-6.7b",
        "target_module": ["query", "value"],
        "num_experts": 4,
        "mnli": {"batch_size": 16, "epoch": 30, "r": 8, "alpha": 8, "max_seqlen": 512, "learning_rate": 5E-04},
        "sst2": {"batch_size": 64, "epoch": 60, "r": 8, "alpha": 8, "max_seqlen": 512, "learning_rate": 5E-04},
        "mrpc": {"batch_size": 32, "epoch": 30, "r": 8, "alpha": 8, "max_seqlen": 512, "learning_rate": 4E-04},
        "cola": {"batch_size": 64, "epoch": 80, "r": 8, "alpha": 8, "max_seqlen": 128, "learning_rate": 5E-04},
        "qnli": {"batch_size": 4, "epoch": 25, "r": 8, "alpha": 8, "max_seqlen": 512, "learning_rate": 4E-04},
        "qqp": {"batch_size": 16, "epoch": 25, "r": 8, "alpha": 8, "max_seqlen": 512, "learning_rate": 5E-04},
        "rte": {"batch_size": 8, "epoch": 80, "r": 8, "alpha": 8, "max_seqlen": 512, "learning_rate": 5E-04},
        "stsb": {"batch_size": 16, "epoch": 40, "r": 8, "alpha": 8, "max_seqlen": 512, "learning_rate": 4E-04},
    }

    # parameters_robert_l = {
    #     "model_checkpoint": "roberta-large",
    #     # "target_module": ["output.dense"],
    #     "target_module": ["query", "value"],
    #     "num_experts": 8,
    #     "mnli": {"batch_size": 16, "epoch": 10, "r": 8, "alpha": 16, "max_seqlen": 128, "learning_rate": 3E-04},
    #     "sst2": {"batch_size": 4, "epoch": 10, "r": 8, "alpha": 16, "max_seqlen": 128, "learning_rate": 4E-04},
    #     "mrpc": {"batch_size": 16, "epoch": 20, "r": 8, "alpha": 16, "max_seqlen": 128, "learning_rate": 3E-04},
    #     "cola": {"batch_size": 32, "epoch": 40, "r": 8, "alpha": 8, "max_seqlen": 128, "learning_rate": 2E-04},
    #     "qnli": {"batch_size": 4, "epoch": 10, "r": 8, "alpha": 16, "max_seqlen": 128, "learning_rate": 2E-04},
    #     "qqp": {"batch_size": 4, "epoch": 20, "r": 8, "alpha": 16, "max_seqlen": 128, "learning_rate": 3E-04},
    #     "rte": {"batch_size": 8, "epoch": 20, "r": 8, "alpha": 16, "max_seqlen": 128, "learning_rate": 4E-04},
    #     "stsb": {"batch_size": 8, "epoch": 30, "r": 8, "alpha": 16, "max_seqlen": 128, "learning_rate": 2E-04},
    # }

    for task in GLUE_TASKS:
        task = "stsb"
        result_dict = {}
        result, log = train(task, parameters_robert_b)
        result_dict["result"] = result
        result_dict["log"] = log

        model_name = parameters_robert_b["model_checkpoint"]
        with open(f"glue_lst_{task}_{model_name}.pickle", 'wb') as f:
            pickle.dump(result_dict, f)

        # exit(0)