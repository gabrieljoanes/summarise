# utils/summarizer.py
import json
import tiktoken
import openai
import os

def count_tokens(text, model_name="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))

def compute_token_reduction(original, summarized):
    return round((1 - summarized / original) * 100, 1) if original else 0.0

def summarize_text(text, model, client, max_target_tokens=100):
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "system",
            "content": "Résume le texte ci-dessous de façon concise sans perdre les faits importants. Garde un ton neutre et journalistique."
        }, {
            "role": "user",
            "content": text
        }],
        max_tokens=max_target_tokens,
        temperature=0.3
    )
    result = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    return result, prompt_tokens, completion_tokens

def estimate_cost(prompt_tokens, completion_tokens, config):
    return (prompt_tokens * config["prompt"] + completion_tokens * config["completion"]) / 1000

def load_examples_from_json_or_jsonl(file):
    if file.name.endswith(".jsonl"):
        return [json.loads(line) for line in file.read().decode("utf-8").splitlines()]
    elif file.name.endswith(".json"):
        return json.load(file)
    else:
        raise ValueError("Unsupported file type")

def save_summarized_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

def log_token_metrics(index, original, target, prompt, completion):
    reduction = compute_token_reduction(original, completion)
    return f"[#{index}] Original: {original} → Target: {target} → Prompt: {prompt} / Completion: {completion} → ↓ {reduction}%"
