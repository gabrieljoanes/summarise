import streamlit as st
import json
import io
from utils.summary import summarize_with_ratio

# Configure Streamlit page
st.set_page_config(page_title="Adjustable Summariser", layout="wide")
st.title("📝 Adjustable JSON Summariser")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a JSON file", type="json")

# Summary length selector
summary_ratio = st.slider(
    "✂️ Choose summary length as a percentage of original word count",
    min_value=10, max_value=90, value=20, step=5
) / 100.0  # Convert to float for use in summarization

# Processing and display logic
if uploaded_file:
    try:
        raw_data = json.load(uploaded_file)
        summarized_results = []

        for entry in raw_data:
            input_text = entry.get("input", "")
            transition = entry.get("output", "")
            summaries = summarize_with_ratio(input_text, summary_ratio)

            summarized_results.append({
                "input": input_text,
                "summaries": summaries,
                "original_output": transition
            })

        st.success(f"✅ Successfully processed {len(summarized_results)} entries.")

        # Display each result
        for idx, result in enumerate(summarized_results, start=1):
            st.markdown(f"---\n### 🧾 Entry {idx}")
            st.markdown("#### 🔹 Original Input")
            st.text(result["input"])

            st.markdown("#### ✂️ Summarized Sections")
            for i, section in enumerate(result["summaries"], start=1):
                st.markdown(f"**Section {i}:** {section}")

            st.markdown("#### 🔄 Original Transition Output")
            st.text(result["original_output"])

        # JSON download
        download_str = json.dumps(summarized_results, ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 Download Summarized JSON",
            data=io.StringIO(download_str),
            file_name="summarized_output.json",
            mime="application/json"
        )

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
