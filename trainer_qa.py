import pickle

task = "cola"

model_name = "opt-6.7b"
with open(f"glue_lst_{task}_{model_name}.pickle", "rb") as f:
    loaded_dict = pickle.load(f)
# print(loaded_dict)

values = []
for elem in loaded_dict["log"]:
    # print(elem)
    # if 'loss' in elem.keys():
    #     values.append(elem['loss'])"eval_matthews_correlation"
    if task == "cola":
        values.append(elem['eval_matthews_correlation'])
    elif task == "stsb":
        values.append(elem['eval_pearson'])
    else:
        values.append(elem['eval_accuracy'])

max_lora = max(values)
print(max_lora)

# model_name = "/home/zzx/pythonproject/LLM/huggingface-demos/experiments/faster_generation/opt-1.3b"
with open(f"glue_lst_{task}_{model_name}.pickle", "rb") as f:
    loaded_dict = pickle.load(f)
# print(loaded_dict)

values = []
for elem in loaded_dict["log"]:
    # print(elem)
    # if 'loss' in elem.keys():
    #     values.append(elem['loss'])eval_accuracy
    if task == "cola":
        values.append(elem['eval_matthews_correlation'])
    elif task == "stsb":
        values.append(elem['eval_pearson'])
    else:
        values.append(elem['eval_accuracy'])

max_moelora = max(values)
print(max_moelora)