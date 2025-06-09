# summarizer.py

from openai import OpenAI
import tiktoken

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def summarize_text(input_text: str, client: OpenAI, model: str, reduction: float) -> tuple[str, int]:
    original_tokens = count_tokens(input_text, model)
    max_target_tokens = int(original_tokens * (1 - reduction))

    messages = [
        {"role": "system", "content": "You are a summarizer. Summarize this input text proportionally to the given token target."},
        {"role": "user", "content": f"Summarize the following text to approximately {max_target_tokens} tokens:\n\n{input_text}"},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3
    )

    summary = response.choices[0].message.content.strip()
    summarized_tokens = count_tokens(summary, model)
    return summary, summarized_tokens
