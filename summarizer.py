import os
import tiktoken
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Estimate token count for a given text using the specified OpenAI model.

    Args:
        text (str): Input text.
        model (str): OpenAI model name.

    Returns:
        int: Token count.
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def summarize_text(text: str, shrink_percent: int, model: str = "gpt-3.5-turbo") -> tuple[str, int, int]:
    """
    Summarize text using OpenAI's Chat API, reducing length by a specified percentage.

    Args:
        text (str): Input text to summarize.
        shrink_percent (int): Desired reduction percentage (e.g., 20 for 20% shorter).
        model (str): OpenAI model name.

    Returns:
        tuple[str, int, int]: Summary text, number of prompt tokens, number of completion tokens.
    """
    if not text.strip():
        return "", 0, 0

    total_tokens = count_tokens(text, model)
    target_tokens = int(total_tokens * (1 - shrink_percent / 100))

    prompt = (
        f"Tu es un assistant journaliste. Résume le texte suivant en français de manière informative et neutre. "
        f"Réduis la longueur d'environ {shrink_percent}% tout en gardant l'essentiel.\n\nTexte:\n{text}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    content = response.choices[0].message.content.strip()
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    return content, prompt_tokens, completion_tokens
