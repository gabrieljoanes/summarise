# app.py

import streamlit as st
import json
import jsonlines
import os
from openai import OpenAI
from summarizer import summarize_text, count_tokens
from datetime import datetime

st.set_page_config(page_title="📉 Input Summarizer", layout="wide")
st.title("📉 Input Summarizer Tool")

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("No API key found in secrets. Please add OPENAI_API_KEY.")
    st.stop()

client = OpenAI(api_key=api_key)

model = st.radio("Select model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"], horizontal=True)
reduction_percent = st.slider("Target reduction percentage", 10, 90, 50)
reduction_ratio = reduction_percent / 100

uploaded_file = st.file_uploader("Upload a .json or .jsonl file with 'input' fields", type=["json", "jsonl"])
no_limit = st.checkbox("🔓 Treat all examples (no max limit)")
max_examples = None if no_limit else st.slider("Max examples to treat", 1, 100, 10)

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[1]
    examples = []

    if file_ext == ".jsonl":
        with jsonlines.Reader(uploaded_file) as reader:
            for row in reader:
                if "input" in row:
                    examples.append(row)
    else:
        data = json.load(uploaded_file)
        examples = [ex for ex in data if "input" in ex]

    st.success(f"Loaded {len(examples)} examples with 'input' field.")

    if st.button("🚀 Run Summarization"):
        summarized_results = []
        total_cost = 0
        cost_per_1k = {"gpt-3.5-turbo": 0.0015, "gpt-4": 0.03, "gpt-4-turbo": 0.01}

        progress = st.progress(0)
        display_limit = len(examples) if no_limit else min(max_examples, len(examples))

        for i, ex in enumerate(examples[:display_limit]):
            input_text = ex["input"]
            original_tokens = count_tokens(input_text, model)
            try:
                summary, summarized_tokens = summarize_text(input_text, client, model, reduction_ratio)
            except Exception as e:
                summary = f"[ERROR: {str(e)}]"
                summarized_tokens = 0

            cost = (original_tokens + summarized_tokens) * cost_per_1k[model] / 1000
            total_cost += cost

            summarized_results.append({
                "input": input_text,
                "output": summary,
                "original_tokens": original_tokens,
                "summary_tokens": summarized_tokens,
                "reduction_pct": round(100 * (1 - summarized_tokens / original_tokens), 1) if original_tokens > 0 else 0,
                "estimated_cost": round(cost, 4)
            })

            progress.progress((i + 1) / display_limit)

        st.success(f"✅ Done summarizing {display_limit} examples.")
        st.markdown(f"### 💰 Total estimated cost: **${total_cost:.4f}**")

        st.markdown("### 🔍 Detailed Results")
        for i, item in enumerate(summarized_results):
            st.markdown(f"**Ex {i+1}** — 🧮 Original: {item['original_tokens']} → ✂️ Summary: {item['summary_tokens']} tokens (**-{item['reduction_pct']}%**), 💵 ${item['estimated_cost']}")
            st.text_area("Original", item['input'], height=100)
            st.text_area("Summary", item['output'], height=100)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_filename = f"summarized_output_{timestamp}.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(summarized_results, f, ensure_ascii=False, indent=2)

        with open(output_filename, "rb") as f:
            st.download_button("📥 Download Results", data=f, file_name=output_filename, mime="application/json")
