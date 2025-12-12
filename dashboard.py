import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =====================
# Load JSON
# =====================
with open("final_payload.json") as f:
    data = json.load(f)

transcripts = data["transcripts"]
interview = data["interview_scores"]
scores = interview["reviewChecklistResult"]["interviews"]["scores"]
overview = interview["scoresOverview"]

# =====================
# Prepare DataFrame
# =====================
rows = []
for s in scores:
    qid = f"question_{s['id']}"
    t = transcripts[qid]

    rows.append({
        "Question ID": f"Q{s['id']}",
        "Question": t["question"],
        "Transcript": t["transcript"],
        "STT Confidence (%)": t["stt_confidence"],
        "Interview Score (0–4)": s["score"],
        "Reason": s["reason"]
    })

df = pd.DataFrame(rows)

# =====================
# Dashboard UI
# =====================
st.set_page_config(layout="wide")
st.title("AI Interview Assessment Dashboard")
st.caption("Whisper Speech-to-Text + Groq LLM Evaluation")

# ---- Summary Cards ----
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Score", overview["total"])
col2.metric("Interview Avg", overview["interview"])
col3.metric("Interview Normalized", overview["interview_normalized"])
col4.metric("Decision", interview["decision"])

st.divider()

# ---- Charts ----
c1, c2 = st.columns(2)

with c1:
    st.subheader("Interview Score per Question")
    st.bar_chart(df.set_index("Question ID")["Interview Score (0–4)"])

with c2:
    st.subheader("STT Confidence per Question")
    st.bar_chart(df.set_index("Question ID")["STT Confidence (%)"])

st.divider()

# ---- Interactive Detail ----
st.subheader("Detail Interview per Question")

selected_q = st.selectbox(
    "Pilih Question",
    df["Question ID"]
)

row = df[df["Question ID"] == selected_q].iloc[0]

st.markdown(f"### {row['Question ID']} — Question")
st.write(row["Question"])

st.markdown("### Transcript")
st.write(row["Transcript"])

st.markdown("### Scores")
st.write(f"**STT Confidence:** {row['STT Confidence (%)']} %")
st.write(f"**Interview Score:** {row['Interview Score (0–4)']} / 4")

st.markdown("### Assessment Reason")
st.info(row["Reason"])
