# Basics
import os
import yaml
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Hugging face
from huggingface_hub import login

# Model
import torch
from transformers import HybridCache
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# Evaluation
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import metrics.bleu as bleu

# Visualization
import utils.vis as vis


# Params
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
  inputs = tokenizer(prompt, return_tensors="pt").to(device)  # pt = pytorch tensor
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


# Self-BLEU score: https://g.co/gemini/share/c3f3a3318026, clipped unigrams / unigrams in ref sentences
def compute_self_bleu(generated_sentences):
    total_bleu_score = 0
    num_sentences = len(generated_sentences)
    for i, hypothesis in enumerate(generated_sentences):
        references = [generated_sentences[j] for j in range(num_sentences) if j != i]
        bleu_scores = [sentence_bleu([ref.split()], hypothesis.split()) for ref in references]
        total_bleu_score += sum(bleu_scores) / len(bleu_scores)
    return total_bleu_score / num_sentences


def compute_corpus_self_bleu(generated_sentences):
    total_bleu_score = 0
    num_sentences = len(generated_sentences)
    for i, hypothesis in enumerate(generated_sentences):
        references = [generated_sentences[j] for j in range(num_sentences) if j != i]
        references_for_bleu = [ref.split() for ref in references]
        hypothesis_for_bleu = hypothesis.split()
        bleu_scores = corpus_bleu([references_for_bleu], [hypothesis_for_bleu])
        total_bleu_score += bleu_scores
    return total_bleu_score / num_sentences


if __name__ == '__main__':
    # get hugging face token
    with open("api/api.yaml", "r") as f:
        api_cfg = yaml.safe_load(f)
        hf_tok = api_cfg["hf_tok"]

    # login
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_model_safely(AutoTokenizer, base_model_name)
    model = load_model_safely(
        AutoModelForCausalLM,
        base_model_name,
        device_map=device,
        torch_dtype=dtype,
    )

    # ## Q1. Chat template comparison
    question = "Please tell me about the key differences between supervised learning and unsupervised learning. Answer in 200 words."
    # # 1.1 Without chat template
    # response_without_template = generate_text_from_prompt(question, tokenizer, model)
    # response_without_template = response_without_template.split('words.')[-1]
    # print("========== Response w/o template ==========\n", response_without_template)
    # # coherent score: https://g.co/gemini/share/84559c5c9edb
    # wo_score = calculate_coherence(question, response_without_template, scoring_model, scoring_tokenizer)
    # print(f"========== Coherence score w/o template : {wo_score:.4f}  ==========")
    #
    # # 1.2 With chat template
    chat = [{"role": "user", "content": question},]
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
    # round = 0
    # print("Chatbot: Hello! How can I assist you today? (Type 'exit' to quit)")
    #
    # while True:
    #     user_input = input("You: ")
    #     if user_input.lower() == "exit":
    #         print("Chatbot: Goodbye!")
    #         break
    #
    #     round += 1
    #     chat_history.append({"role": "user", "content": user_input})
    #     chat_template_format_prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    #
    #     # 2.1 Observe the prompt with chat template format that was inputted to the model in the current round
    #     print(
    #         f"=== Prompt with chat template format inputted to the model on round {round} ===\n{chat_template_format_prompt}")
    #     print(f"===============================================")
    #
    #     # Tokenization
    #     inputs = tokenizer(chat_template_format_prompt, return_tensors="pt").to(device)
    #
    #     # Get logits instead of directly generating
    #     with torch.no_grad():
    #         outputs_p = model(**inputs)  # without generate
    #     logits = outputs_p.logits  # logits of the model output (raw scores before softmax)
    #     last_token_logits = logits[:, -1, :]  # logits of the last generated token
    #
    #     # Apply softmax to get probabilities
    #     probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
    #
    #     # Get top-k tokens (e.g., 10)
    #     top_probs, top_indices = torch.topk(probs, k=10)
    #
    #     # Convert to numpy for plotting
    #     top_probs = top_probs.cpu().squeeze().numpy()
    #     top_indices = top_indices.cpu().squeeze().numpy()
    #     top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
    #
    #     # Plot probability distribution
    #     plt.figure(figsize=(10, 5))
    #     sns.barplot(x=top_probs, y=top_tokens, palette="coolwarm")
    #     plt.xlabel("Probability")
    #     plt.ylabel("Token")
    #     plt.title("Top Token Probabilities for Next Word")
    #     plt.show()
    #
    #     # Generate response
    #     outputs = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    #     response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)  # response contains the input tokens, start from the model output
    #     print(f"Chatbot: {response}")
    #     chat_history.append({"role": "assistant", "content": response})

    ## Q3. Tokenization a sentence
    sentence = "I love taking a Machine Learning course by Professor Hung-yi Lee, What about you?"
    # inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).to(device) # pt = pytorch tensor
    # token_ids = inputs.input_ids[0]
    # for t_id in token_ids:
    #     t = tokenizer.decode(t_id, skip_special_tokens=True)  # decode token ids back to tokens, i.e., words or subwords
    #     print(f"Token: {t} --- Token index: {t_id}")  # print the token and its token index (token id)

    ## Q4. Autoregressive generation
    prompt = f"Generate a paraphrase of the sentence 'Professor Hung-yi Lee is one of the best teachers in the domain of machine learning'. Just response with one sentence."
    # inputs = tokenizer(prompt, return_tensors="pt")
    #
    # # initialize kv cache
    # max_generation_tokens = 30
    # top_k = 200  # set K for top-k sampling, when k=1, greedy decoding
    # top_p = 0.99  # set P for nucleus sampling, when p=0, greedy decoding, equals to k=1
    # kv_cache = HybridCache(config=model.config, max_batch_size=1, max_cache_len=max_generation_tokens, device=device,  # TODO: what is kv cache?
    #                        dtype=torch.float16)
    # input_ids = inputs.input_ids.to(device)
    # attention_mask = inputs.attention_mask.to(device)
    # cache_position = torch.arange(attention_mask.shape[1], device=device)
    #
    # # define generation
    # generated_sentences_top_k = []
    # generated_sentences_top_p = []
    # generation_params = {
    #     "do_sample": True,  # enable sampling
    #     "max_length": len(input_ids[0]) + max_generation_tokens,  # total length including prompt
    #     "pad_token_id": tokenizer.pad_token_id,  # ensure padding token is set
    #     "eos_token_id": tokenizer.eos_token_id,  # ensure EOS token is set
    #     "bos_token_id": tokenizer.bos_token_id,  # ensure BOS token is set
    #     "attention_mask": attention_mask,  # move attention mask to GPU
    #     "use_cache": True,  # enable kv caching
    #     "return_dict_in_generate": True,  # return generation outputs
    #     "output_scores": False,  # disable outputting scores
    # }
    #
    # # autoregressive generation loop
    # for method in ["top-k", "top-p"]:  # top-k and top-p generation: https://g.co/gemini/share/24cbbc1a6289
    #     for _ in trange(20):
    #         if method == "top-k":
    #             # generate text using the model with top_k
    #             generated_output = model.generate(
    #                 input_ids=input_ids,
    #                 top_k=top_k,
    #                 **generation_params
    #             )
    #         elif method == "top-p":
    #             # generate text using the model with top_p
    #             generated_output = model.generate(
    #                 input_ids=input_ids,
    #                 top_p=top_p,
    #                 top_k=0,
    #                 **generation_params
    #             )
    #             ###################################################################
    #         else:
    #             raise NotImplementedError()
    #         # decode the generated tokens
    #         generated_tokens = generated_output.sequences[0, len(input_ids[0]):]
    #         decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    #
    #         # Combine the prompt with the generated text
    #         sentence = decoded_text.replace(" ,", ",").replace(" 's", "'s").replace(" .", ".").strip()
    #
    #         # Append the generated sentence to the appropriate list
    #         if method == "top-k":
    #             generated_sentences_top_k.append(sentence)
    #         else:
    #             generated_sentences_top_p.append(sentence)
    #
    # # calculate BLEU score for top-k and top-p
    # # | Method | Parameter | nltk BLEU  | Self-coded BLEU |
    # # |--------|-----------|------------|-----------------|
    # # | top-k  | k=2       | 0.2389     | 0.8292         |
    # # | top-k  | k=200     | 0.0779     | 0.5184          |
    # # | top-p  | p=0.6     | 0.4994     | 0.9248          |
    # # | top-p  | p=0.999   | 0.0959     | 0.5221          |
    #
    # # nltk self-BLEU score
    # bleu_score_k = compute_corpus_self_bleu(generated_sentences_top_k)
    # bleu_score_p = compute_corpus_self_bleu(generated_sentences_top_p)
    # print(f"nltk self-BLEU Score for top_k (k={top_k}): {bleu_score_k:.4f}")
    # print(f"nltk self-BLEU Score for top_p (p={top_p}): {bleu_score_p:.4f}")
    #
    # # self coded self-BLEU score
    # bleu_score_k = bleu.calculate_self_bleu(generated_sentences_top_k, max_n=4)
    # bleu_score_p = bleu.calculate_self_bleu(generated_sentences_top_p, max_n=4)
    # print(f"Self-coded self-BLEU Score for top_k (k={top_k}): {bleu_score_k:.4f}")
    # print(f"Self-coded self-BLEU Score for top_p (p={top_p}): {bleu_score_p:.4f}")

    ## Q5. T-SNE visualization
    sentences = [
        "I ate a fresh apple.",  # Apple (fruit)
        "Apple released the new iPhone.",  # Apple (company)
        "I peeled an orange and ate it.",  # Orange (fruit)
        "The Orange network has great coverage.",  # Orange (telecom)
        "Microsoft announced a new update.",  # Microsoft (company)
        "Banana is my favorite fruit.",  # Banana (fruit)
    ]

    # # tokenize and move to device
    # model.to(device)
    # inputs = tokenizer(sentences, return_tensors="pt", padding=True)  # pad all sentences in a batch with 0
    # inputs = inputs.to(device)
    #
    # # get hidden states
    # with torch.no_grad():
    #     outputs = model(**inputs, output_hidden_states=True)  # [layers=27, batch_size=6, sequence_length=9, hidden_dimension=2304]
    # hidden_states = outputs.hidden_states[-1]  # extract last layer embeddings
    #
    # # compute sentence-level embeddings (mean pooling)
    # sentence_embeddings = hidden_states.mean(dim=1).cpu().numpy()
    #
    # # word-level embeddings
    # # a = hidden_states[0, 4, :]
    # # b = hidden_states[1, 0, :]
    # # c = hidden_states[2, 3, :]
    # # d = hidden_states[3, 1, :]
    # # e = hidden_states[4, 0, :]
    # # f = hidden_states[5, 0, :]
    # # word_embeddings = torch.stack([a, b, c, d, e, f]).cpu().numpy()
    #
    # # words to visualize
    # word_labels = [
    #     "Apple (fruit)", "Apple (company)",
    #     "Orange (fruit)", "Orange (telecom)",
    #     "Microsoft (company)", "Banana (fruit)"
    # ]
    # vis.tsne_vis(sentence_embeddings, word_labels)

    ## Q6. Vis attention weight
    prompt = "Google "
    input_ids = tokenizer(prompt, return_tensors="pt")  # Tokenize the input prompt
    next_token_id = input_ids.input_ids.to("cuda")  # Move input token ids to GPU
    attention_mask = input_ids.attention_mask.to("cuda")  # Move attention mask to GPU
    cache_position = torch.arange(attention_mask.shape[1], device="cuda")  # Position for the KV cache

    # Set the number of tokens to generate and other parameters
    generation_tokens = 20  # Limit for visualization (number of tokens to generate)
    total_tokens = generation_tokens + next_token_id.size(1) - 1  # Total tokens to handle
    layer_idx = 10  # Specify the layer index for attention visualization
    head_idx = 7  # Specify the attention head index to visualize

    # KV cache setup for caching key/values across time steps
    from transformers.cache_utils import HybridCache

    kv_cache = HybridCache(config=model.config, max_batch_size=1, max_cache_len=total_tokens, device="cuda",
                           dtype=torch.float16)

    generated_tokens = []  # List to store generated tokens
    attentions = None  # Placeholder to store attention weights

    num_new_tokens = 0  # Counter for the number of new tokens generated
    model.eval()  # Set the model to evaluation mode

    # Generate tokens and collect attention weights for visualization
    for num_new_tokens in range(generation_tokens):
        with torch.no_grad():  # Disable gradients during inference for efficiency
            # Pass the input through the model to get the next token prediction and attention weights
            outputs = model(
                next_token_id,
                attention_mask=attention_mask,
                cache_position=cache_position,
                use_cache=True,  # Use the KV cache for efficiency
                past_key_values=kv_cache,  # Provide the cached key-value pairs for fast inference
                output_attentions=True  # Enable the extraction of attention weights
            )

        ######################## TODO (Q6.1 ~ 6.4) ########################
        ### You can refer to https://huggingface.co/docs/transformers/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput.attentions to see the structure of model output attentions
        # Get the logits for the last generated token from outputs
        logits = 1
        # Extract the attention scores from the model's outputs
        attention_scores = 1
        ###################################################################

        # Extract attention weights for the specified layer and head
        last_layer_attention = attention_scores[layer_idx][0][head_idx].detach().cpu().numpy()

        # If it's the first generated token, initialize the attentions array
        if num_new_tokens == 0:
            attentions = last_layer_attention
        else:
            # Append the current attention weights to the existing array
            attentions = np.append(attentions, last_layer_attention, axis=0)

        # Choose the next token to generate based on the highest probability (logits)
        next_token_id = logits.argmax(dim=-1)
        generated_tokens.append(next_token_id.item())  # Add the token ID to the generated tokens list

        # Update the attention mask and next token ID for the next iteration
        attention_mask = torch.cat([attention_mask, torch.ones(1, 1, device="cuda")],
                                   dim=-1)  # Add a new attention mask for the generated token
        next_token_id = next_token_id.unsqueeze(0)  # Convert the token ID to the required shape

        # Update the KV cache with the new past key-values
        kv_cache = outputs.past_key_values
        cache_position = cache_position[-1:] + 1  # Update the cache position for the next iteration

    # Decode the generated tokens into human-readable text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = prompt + generated_text  # Combine the prompt with the generated text

    # Tokenize all the generated text (prompt + generated)
    tokens = tokenizer.tokenize(full_text)

    # Plot the attention heatmap for the last generated token
    vis.plot_attention(attentions, tokens, title=f"Attention Weights for Generated Token of Layer {layer_idx}")