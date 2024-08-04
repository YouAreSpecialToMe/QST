from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from modeling_llama_qst import QSTLlamaForCausalLM, LlamaForCausalLM
import torch
from QSTConfig import QSTConfig
from peft.tuners.lora import LoraLayer
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.generation.utils')


# print(user_input)
# exit(0)

compute_dtype = torch.float32
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-chat-hf",
    load_in_4bit=True,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
    torch_dtype=torch.float32,
)

setattr(model, 'model_parallel', True)
setattr(model, 'is_parallelizable', True)

model.config.torch_dtype = torch.float32
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-70b-chat-hf",
    padding_side="right",
    use_fast=False,  # Fast tokenizer giving issues.
    tokenizer_type='llama'
    # Needed for HF name change
)

model.config.pad_token_id = 0
tokenizer.add_special_tokens({
    "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
    "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
    "unk_token": tokenizer.convert_ids_to_tokens(
        model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
    ),
})

config = AutoConfig.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
config.rope_theta = 10000.0
qst_config = QSTConfig(
    add_layer_norm_before_adapter=False,
    add_layer_norm_after_adapter=True,
    r=16,
    alpha_r=16,
    dropout=0.1,
    activation="swish",
    fan_in_fan_out=False,
    peft_hidden_size=16
)

model = QSTLlamaForCausalLM(model, config, qst_config)
# print(model.hf_device_map)

model.config.pad_token_id = 0
setattr(model, 'model_parallel', True)
setattr(model, 'is_parallelizable', True)
model.config.torch_dtype = torch.float32

model.load_qst_state("YourPath/QST-70B-checkpoint/")
# model.config.use_cache = False

for name, module in model.named_modules():
    if isinstance(module, LoraLayer):
        module = module.to(torch.float32)
    if 'norm' in name:
        module = module.to(torch.float32)
    if 'lm_head' in name or 'embed_tokens' in name:
        if hasattr(module, 'weight'):
            if module.weight.dtype == torch.float32:
                module = module.to(torch.float32)

# prompt
chat_history_ids = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. ### Human: Got any creative ideas for a 10 year old’s birthday? 
### Assistant: Of course! Here are some creative ideas for a 10-year-old's birthday party:  
1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises. 
2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions. 
3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars. 
4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors. 
5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants. 
6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen. 
7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges. 
8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors. 
Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!"""

# do not have the history information
while True:

    user_input = input("You: ")
    user_input = chat_history_ids + "\n### Human: " + user_input + "\n### Assistant: "

    new_user_input_ids = tokenizer.encode(user_input, return_tensors='pt').to("cuda:3")

    # if chat_history_ids is not None:
    #     bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    # else:
    bot_input_ids = new_user_input_ids


    outputs = model.generate(bot_input_ids, do_sample=True,
                                      top_p=0.9, max_length=1024)

    bot_response = tokenizer.decode(outputs[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    print(f"QST-70B: {bot_response}")
