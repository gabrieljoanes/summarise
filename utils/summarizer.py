import openai
import tiktoken

def num_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def summarize_examples(examples, model="gpt-3.5-turbo", ratio=0.6, max_count=100):
    summarized = []
    logs = []

    for i, ex in enumerate(examples[:max_count]):
        original = ex.get("input", "")
        target_len = int(num_tokens(original, model) * ratio)

        prompt = f"Résume le texte suivant de manière informative, sans changer les faits, pour réduire le nombre de tokens à environ {target_len} tokens:\n\n{original}"

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Tu es un assistant de résumé pour des textes journalistiques."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=target_len + 50  # allow margin
            )
            summary = response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            summary = f"[ERREUR: {e}]"

        summarized.append({
            "input": summary,
            "output": ex.get("output", "")
        })

        logs.append({
            "original_tokens": num_tokens(original, model),
            "summary_tokens": num_tokens(summary, model)
        })

    return summarized, logs

def estimate_tokens(input_toks, output_toks, model):
    rates = {
        "gpt-3.5-turbo": {"prompt": 0.001, "completion": 0.002},
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03}
    }
    if model not in rates:
        return 0.0
    p = rates[model]["prompt"]
    c = rates[model]["completion"]
    return (input_toks * p + output_toks * c) / 1000
