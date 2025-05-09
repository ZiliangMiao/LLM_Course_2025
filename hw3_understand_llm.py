# hugging face
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from huggingface_hub import login
import os

# model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification


# Mode
LOCAL_MODE = True
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
base_model_path = {
    "Qwen/Qwen3-8B": "/home/ziliang/Projects/LLM/pretrained models/models--Qwen--Qwen3-8B",
}

# 0: cache, 1: local, 2: web
model_place_map = {
    "google/gemma-2-2b-it": 0,
    "google/gemma-3-4b-pt": 0,
    "google/gemma-3-4b-it": 0,
    "meta-llama/Meta-Llama-3-8B": 0,
    "Qwen/Qwen3-8B": 1,
    "cross-encoder/ms-marco-MiniLM-L-6-v2": 0,
}

# Load model
def load_model_safely(model_class, model_name, **kwargs):
    model_place = model_place_map[model_name]
    match model_place:
        case 0:  # cache
            print(f"从缓存加载 {model_name}")
            return model_class.from_pretrained(model_name, local_files_only=True, **kwargs)
        case 1:  # local
          print(f"从本地加载 {model_name}")
          return model_class.from_pretrained(base_model_path[model_name], **kwargs)
        case 2:  # web
          print(f"从网络加载 {model_name}")
          return model_class.from_pretrained(model_name, **kwargs)


# Coherence
def calculate_coherence(question, answer, scoring_model, scoring_tokenizer):
    features = scoring_tokenizer([question], [answer], padding=True, truncation=True, return_tensors="pt")
    scoring_model.eval()
    with torch.no_grad():
        scores = scoring_model(**features).logits.squeeze().item()
    return scores


# Generate response
def generate_text_from_prompt(prompt, tokenizer, model):
  """
  generate the output from the prompt.
  param:
    prompt (str): the prompt inputted to the model
    tokenizer   : the tokenizer that is used to encode / decode the input / output
    model       : the model that is used to generate the output

  return:
    the response of the model
  """
  print("========== Prompt inputted to the model ==========\n", prompt)
  inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # pt = pytorch tensor
  outputs = model.generate(
      **inputs,  # **inputs将字典解包, 包含`input_ids`和`attention_mask`
      max_new_tokens=1000,
      do_sample=False,
      pad_token_id=tokenizer.eos_token_id
  )

  if outputs is not None and len(outputs) > 0:
      generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
      return generated_text
  else:
      return "Empty Response"


if __name__ == '__main__':
    # get hugging face token from config file
    with open("api/api.yaml", "r") as f:
        api_cfg = yaml.safe_load(f)
        hf_tok = api_cfg["hf_tok"]

    # 根据模式决定是否登录
    if not LOCAL_MODE:
        try:
            print("登录 Hugging Face...")
            login(hf_tok)
            print("登录成功!")
        except Exception as e:
            print(f"登录失败: {e}")
    else:
        print("离线模式，跳过登录")

    # load scoring model and tokenizer
    scoring_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    scoring_model = load_model_safely(AutoModelForSequenceClassification, scoring_model_name)
    scoring_tokenizer = load_model_safely(AutoTokenizer, scoring_model_name)

    # load base model and tokenizer
    base_model_name = "google/gemma-2-2b-it"  # ["google/gemma-2-2b-it", "google/gemma-3-4b-pt", "google/gemma-3-4b-it", "Qwen/Qwen3-8B", "meta-llama/Meta-Llama-3-8B"]
    dtype = torch.float16
    tokenizer = load_model_safely(AutoTokenizer, base_model_name)
    model = load_model_safely(
        AutoModelForCausalLM,
        base_model_name,
        device_map="cuda",
        torch_dtype=dtype,
    )

    # ## Q1. Chat template comparison
    # question = "Please tell me about the key differences between supervised learning and unsupervised learning. Answer in 200 words."
    # # 1.1 Without chat template
    # response_without_template = generate_text_from_prompt(question, tokenizer, model)
    # response_without_template = response_without_template.split('words.')[-1]
    # print("========== Response w/o template ==========\n", response_without_template)
    # # coherent score: https://g.co/gemini/share/84559c5c9edb
    # wo_score = calculate_coherence(question, response_without_template, scoring_model, scoring_tokenizer)
    # print(f"========== Coherence score w/o template : {wo_score:.4f}  ==========")
    #
    # # 1.2 With chat template
    # chat = [{"role": "user", "content": question},]
    # prompt_with_template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    # response_with_template = generate_text_from_prompt(prompt_with_template, tokenizer, model)
    # response_with_template = response_with_template.split('model\n')[-1].strip('\n').strip()
    # print("========== Response w/ template ==========\n", response_with_template)
    # w_score = calculate_coherence(question, response_with_template, scoring_model, scoring_tokenizer)
    # print(f"========== Coherence score w/ template : {w_score:.4f}  ==========")

    ## Q2. Multi-turn conversation
    # 1st: "Name a color in a rainbow, please just answer in a word without any emoji."
    # 2nd: "Thats great! Now, could you tell me another color that I can find in a rainbow?"
    # 3rd: "Could you continue and name yet another color from the rainbow?"
    chat_history = []
    round = 0
    print("Chatbot: Hello! How can I assist you today? (Type 'exit' to quit)")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        round += 1
        chat_history.append({"role": "user", "content": user_input})
        chat_template_format_prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)

        # 2.1 Observe the prompt with chat template format that was inputted to the model in the current round
        print(
            f"=== Prompt with chat template format inputted to the model on round {round} ===\n{chat_template_format_prompt}")
        print(f"===============================================")

        # Tokenization
        inputs = tokenizer(chat_template_format_prompt, return_tensors="pt").to("cuda")

        # Get logits instead of directly generating
        with torch.no_grad():
            outputs_p = model(**inputs)  # without generate
        logits = outputs_p.logits  # logits of the model output (raw scores before softmax)
        last_token_logits = logits[:, -1, :]  # logits of the last generated token

        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(last_token_logits, dim=-1)

        # Get top-k tokens (e.g., 10)
        top_probs, top_indices = torch.topk(probs, k=10)

        # Convert to numpy for plotting
        top_probs = top_probs.cpu().squeeze().numpy()
        top_indices = top_indices.cpu().squeeze().numpy()
        top_tokens = [tokenizer.decode([idx]) for idx in top_indices]

        # Plot probability distribution
        plt.figure(figsize=(10, 5))
        sns.barplot(x=top_probs, y=top_tokens, palette="coolwarm")
        plt.xlabel("Probability")
        plt.ylabel("Token")
        plt.title("Top Token Probabilities for Next Word")
        plt.show()

        # Generate response
        outputs = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)  # response contains the input tokens, start from the model output
        print(f"Chatbot: {response}")
        chat_history.append({"role": "assistant", "content": response})
