# summarizer.py
import tiktoken
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def count_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def summarize_text(text, shrink_percent, model="gpt-3.5-turbo"):
    max_input_tokens = count_tokens(text, model)
    target_tokens = int(max_input_tokens * (1 - shrink_percent / 100))

    prompt = (
        f"Tu es un assistant journaliste. Résume le texte suivant en français de manière informative et neutre. "
        f"Conserve l'essentiel en réduisant la longueur d'environ {shrink_percent}%.\n\nTexte:\n{text}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    summary = response.choices[0].message.content.strip()
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    return summary, prompt_tokens, completion_tokens
