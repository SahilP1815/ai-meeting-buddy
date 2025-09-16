import streamlit as st
import re
import whisper
from transformers import pipeline

# Load models once
@st.cache_resource
def load_models():
    stt_model = whisper.load_model("base")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return stt_model, summarizer

st.title("🤖 AI Meeting Buddy (Prototype)")
st.write("Upload a meeting audio file and get transcript, summary, and action items.")

# Load models
stt_model, summarizer = load_models()

# Upload audio
audio_file = st.file_uploader("🎙️ Upload Meeting Audio (mp3/wav)", type=["mp3", "wav"])

if audio_file:
    # Save temp file
    with open("temp_audio.mp3", "wb") as f:
        f.write(audio_file.read())

    st.subheader("📜 Transcript")
    result = stt_model.transcribe("temp_audio.mp3")
    transcript = result["text"]
    st.text_area("Transcript:", transcript, height=200)

    # --- Step 1: Summarization ---
    st.subheader("📌 Meeting Summary")
    summary = summarizer(transcript, max_length=130, min_length=30, do_sample=False)
    st.write(summary[0]['summary_text'])

    # --- Step 2: Action Item Extraction ---
    st.subheader("✅ Action Items")
    action_pattern = r"(?:\b[A-Z][a-z]+)\s+(?:will|should|must|to)\s+([^.]+)"
    actions = re.findall(action_pattern, transcript)

    if actions:
        for i, act in enumerate(actions, 1):
            st.write(f"**Task {i}:** {act.strip()}")
    else:
        st.write("No clear action items detected.")

    # --- Step 3: Q&A ---
    st.subheader("❓ Ask About Meeting")
    user_question = st.text_input("Example: Who will prepare the report?")
    if user_question:
        if any(word.lower() in transcript.lower() for word in user_question.split()):
            st.write("🔎 Found in transcript context:")
            st.write("..." + transcript[:400] + "...")
        else:
            st.write("❌ Couldn't find relevant info in transcript.")
