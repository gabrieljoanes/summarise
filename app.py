import streamlit as st
import json
import jsonlines
import os
import openai
from utils.summarizer import summarize_examples, estimate_tokens

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="📉 Input Summarizer", page_icon="🧠")

st.title("📉 Résumeur d'exemples JSON/JSONL")

uploaded_file = st.file_uploader("📄 Chargez un fichier `.json` ou `.jsonl`", type=["json", "jsonl"])
model_choice = st.selectbox("🤖 Modèle OpenAI", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])
shrink_ratio = st.slider("🔽 Réduction approximative de l'entrée", 0.1, 0.9, 0.6, step=0.05)
max_examples = st.number_input("🔢 Nombre maximal d'exemples à traiter", min_value=1, value=100)

if st.button("⚙️ Lancer la réduction"):
    if uploaded_file:
        ext = os.path.splitext(uploaded_file.name)[1]
        raw_examples = []

        if ext == ".json":
            raw_examples = json.load(uploaded_file)
        elif ext == ".jsonl":
            with jsonlines.Reader(uploaded_file) as reader:
                raw_examples = list(reader)

        if not raw_examples:
            st.warning("Aucun exemple détecté.")
        else:
            with st.spinner("⏳ Résumé en cours..."):
                summarized, token_logs = summarize_examples(
                    raw_examples,
                    model=model_choice,
                    ratio=shrink_ratio,
                    max_count=max_examples
                )

            total_input = sum(log["original_tokens"] for log in token_logs)
            total_output = sum(log["summary_tokens"] for log in token_logs)
            total_cost = estimate_tokens(total_input, total_output, model_choice)

            st.success("✅ Résumé terminé")

            st.markdown("### 📊 Statistiques globales")
            st.markdown(f"**Original tokens**: {total_input}")
            st.markdown(f"**Résumé tokens**: {total_output}")
            st.markdown(f"**Réduction moyenne**: {100 * (1 - total_output / total_input):.1f}%")
            st.markdown(f"**Coût estimé**: **${total_cost:.4f}**")

            st.markdown("### 📁 Télécharger le résultat")
            output_filename = "résumé_output.json"
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(summarized, f, ensure_ascii=False, indent=2)
            with open(output_filename, "rb") as f:
                st.download_button("📥 Télécharger le fichier résumé", f, file_name=output_filename)

            st.markdown("### 🪵 Détails par exemple")
            for i, log in enumerate(token_logs):
                st.markdown(f"- Exemple {i+1}: {log['original_tokens']} → {log['summary_tokens']} tokens")

    else:
        st.warning("Veuillez d'abord charger un fichier.")
