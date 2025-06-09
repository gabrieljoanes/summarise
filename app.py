# app.py
import streamlit as st
import json
from datetime import datetime
from openai import OpenAI
from utils.summarizer import (
    count_tokens,
    summarize_text,
    estimate_cost,
    compute_token_reduction,
    load_examples_from_json_or_jsonl,
    save_summarized_jsonl,
    log_token_metrics
)

st.set_page_config(page_title="🧠 Input Summarizer", layout="wide")

MODEL_OPTIONS = {
    "gpt-3.5-turbo": {"prompt": 0.001, "completion": 0.002, "max_tokens": 16000},
    "gpt-4": {"prompt": 0.03, "completion": 0.06, "max_tokens": 8192},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03, "max_tokens": 128000},
}

st.title("📉 Input Summarizer for Few-Shot Examples")

uploaded_file = st.file_uploader("📄 Upload JSON or JSONL file with examples", type=["json", "jsonl"])

if uploaded_file:
    examples = load_examples_from_json_or_jsonl(uploaded_file)
    st.success(f"✅ Loaded {len(examples)} examples")

    model_choice = st.selectbox("🤖 Choose GPT model", list(MODEL_OPTIONS.keys()))
    compression = st.slider("📉 Compression Ratio (approx.)", min_value=0.1, max_value=1.0, step=0.05, value=0.5)
    max_examples = st.slider("🔢 Number of examples to process", 1, min(500, len(examples)), 100)

    model_config = MODEL_OPTIONS[model_choice]
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    if st.button("⚡ Run Summarization"):
        summarized = []
        logs = []
        total_prompt = 0
        total_completion = 0
        progress = st.progress(0)

        for idx, ex in enumerate(examples[:max_examples]):
            original_text = ex["input"]
            original_tokens = count_tokens(original_text, model_choice)
            target_tokens = int(original_tokens * compression)

            summary, p_tokens, c_tokens = summarize_text(
                original_text,
                model=model_choice,
                client=client,
                max_target_tokens=target_tokens
            )

            summarized.append({
                "input": summary.strip(),
                "output": ex["output"],
                "meta": {
                    "original_tokens": original_tokens,
                    "summary_tokens": c_tokens,
                    "reduction": compute_token_reduction(original_tokens, c_tokens),
                }
            })

            logs.append(log_token_metrics(idx, original_tokens, target_tokens, p_tokens, c_tokens))

            total_prompt += p_tokens
            total_completion += c_tokens
            progress.progress((idx + 1) / max_examples)

        cost = estimate_cost(total_prompt, total_completion, model_config)
        st.markdown("### 📊 Token & Cost Summary")
        st.markdown(f"**Total prompt tokens:** {total_prompt}")
        st.markdown(f"**Total completion tokens:** {total_completion}")
        st.markdown(f"**Estimated cost:** ${cost:.4f}")

        filename = f"summarized_{datetime.now().strftime('%Y%m%d%H%M%S')}.jsonl"
        save_summarized_jsonl(summarized, filename)

        with open(filename, "rb") as f:
            st.download_button(
                label="📥 Download summarized JSONL",
                data=f,
                file_name=filename,
                mime="application/jsonl"
            )

        st.markdown("### 📄 Token Metrics per Example")
        st.code("\n".join(logs), language="text")
