# app.py
import streamlit as st
import json
import jsonlines
import os
from summarizer import summarize_text, count_tokens

st.set_page_config(page_title="📝 Résumeur d'exemples", page_icon="📝")
st.title("📝 Résumeur d'exemples (JSON / JSONL)")

uploaded_file = st.file_uploader("📂 Téléversez un fichier JSON ou JSONL", type=["json", "jsonl"])

model = st.selectbox("🤖 Modèle à utiliser", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])
shrink_percent = st.slider("🔧 Pourcentage de réduction souhaité", min_value=10, max_value=90, value=50)
no_limit = st.checkbox("⚠️ Traiter tous les exemples sans limite")
limit = None if no_limit else st.number_input("🔢 Nombre max d'exemples à traiter", min_value=1, value=20, step=1)

if uploaded_file:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    examples = []

    if ext == ".jsonl":
        with jsonlines.Reader(uploaded_file) as reader:
            for item in reader:
                examples.append(item)
    elif ext == ".json":
        examples = json.load(uploaded_file)

    st.write(f"📊 {len(examples)} exemples chargés.")

    summarized_examples = []
    total_prompt = 0
    total_summary = 0

    with st.spinner("✂️ Résumés en cours..."):
        for i, ex in enumerate(examples):
            if limit is not None and i >= limit:
                break
            input_text = ex["input"]
            summary, prompt_t, summary_t = summarize_text(input_text, shrink_percent, model)
            summarized_examples.append({"input": input_text, "output": summary})
            total_prompt += prompt_t
            total_summary += summary_t

            st.markdown(f"**Exemple {i + 1}**")
            st.code(summary)
            st.progress((i + 1) / min(limit or len(examples), len(examples)))

    st.success("✅ Résumés terminés")
    st.markdown(f"**🔢 Tokens totaux utilisés**: prompt = {total_prompt}, summary = {total_summary}")
    rate = {"gpt-3.5-turbo": 0.002, "gpt-4": 0.06, "gpt-4-turbo": 0.03}
    cost = (total_prompt + total_summary) / 1000 * rate[model]
    st.markdown(f"💰 **Coût estimé**: ${cost:.4f}")

    output_filename = "output_summarized.json"
    with open(output_filename, "w") as f:
        json.dump(summarized_examples, f, indent=2, ensure_ascii=False)
    with open(output_filename, "rb") as f:
        st.download_button("📥 Télécharger le fichier JSON", f, file_name=output_filename, mime="application/json")
