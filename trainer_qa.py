import pickle

task = "cola"

with open(f"glue_lst_mrpc_roberta-large.pickle", "rb") as f:
    loaded_dict = pickle.load(f)
# print(loaded_dict)

values = []
for elem in loaded_dict["log"]:
    print(elem)
    # if 'loss' in elem.keys():
    #     values.append(elem['loss'])"eval_matthews_correlation"
    if "eval_accuracy" in elem.keys():
        values.append(elem['eval_accuracy'])

max_lora = max(values)
print(max_lora)

with open(f"glue_lora_cola_opt-6.7b.pickle", "rb") as f:
    loaded_dict = pickle.load(f)
# print(loaded_dict)

values = []
for elem in loaded_dict["log"]:
    # print(elem)
    # if 'loss' in elem.keys():
    #     values.append(elem['loss'])eval_accuracy
    if "eval_matthews_correlation" in elem.keys():
        values.append(elem['eval_matthews_correlation'])

max_moelora = max(values)
print(max_moelora)