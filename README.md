# 🤖 AI Meeting Buddy (Prototype)

AI-powered meeting assistant that:
- Transcribes meeting audio (using OpenAI Whisper).
- Summarizes meeting discussions (using BART).
- Extracts action items (who does what).
- Provides simple Q&A about the meeting.

## 🚀 Features
- Upload meeting audio (mp3/wav).
- Automatic transcription.
- Concise meeting summary.
- Task & action item extraction.
- Simple Q&A over transcript.

## 🛠️ Tech Stack
- **Streamlit** – UI framework
- **OpenAI Whisper** – Speech-to-text
- **Hugging Face Transformers (BART)** – Summarization
- **Regex + NLP** – Action item extraction

## 📂 Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/ai-meeting-buddy.git
   cd ai-meeting-buddy
