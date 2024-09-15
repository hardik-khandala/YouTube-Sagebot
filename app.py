import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
import re
from bs4 import BeautifulSoup
from transformers import pipeline
import warnings
import spacy
import requests
import google.generativeai as genai
import io
import numpy as np
from gtts import gTTS
import speech_recognition as sr
from textblob import TextBlob
import pyttsx3
import pytube
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
genai.configure(api_key="AIzaSyAyRBWQ016M7-GJ65NQ9szFk-TkPHCje_U")

engine = pyttsx3.init()
# Function to get video title
def Title(link):
    try:
        res = requests.get(link)
        html_content = res.text
        soup = BeautifulSoup(html_content, "html.parser")
        video_title = soup.find("meta", property="og:title")["content"]
        return video_title
    except Exception as e:
        return f"Error fetching title: {e}"

def fetch_image(link):
    yt = pytube.YouTube(link)
    
    return yt.thumbnail_url

# Preprocess transcript text using spacy
def PreProcess(text, link):
    nlp_spacy = spacy.load("en_core_web_lg")
    title = f"Title or main topic of the video is {Title(link)}"
    doc = nlp_spacy(text)
    tokens = [token.text for token in doc if not token.is_punct and not token.is_stop]
    tokens.append(title)
    return " ".join(tokens)

# QnA model using a transformer pipeline
def QNA(question, context):
    model_name = "deepset/roberta-base-squad2"
    nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)
    QA_input = {"question": question, "context": context}
    try:
        res = nlp(QA_input)
        return res.get("answer", "Sorry, no valid answer found.")
    except Exception as e:
        return f"Error in QnA: {e}"

# Function to extend answer using Google Generative AI
def generate_extended_answer(prompt, max_output_tokens=1000):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_output_tokens
            )
        )
        answer = response.text
        return answer
    except Exception as e:
        return f"Error generating answer: {e}"

# Function to fetch transcript of a YouTube video
def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry["text"] for entry in transcript])
        return transcript_text
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        return str(e)

# Convert markdown to text
def markdown_to_text(markdown):
     # Remove HTML underline tags if any
    text = re.sub(r'<u>(.*?)</u>', r'\1', markdown)
    
    # Convert bold to plain text
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold using **
    text = re.sub(r'__(.*?)__', r'\1', text)     # Bold using __

    # Convert italic to plain text
    text = re.sub(r'\*(.*?)\*', r'\1', text)     # Italic using *
    text = re.sub(r'_(.*?)_', r'\1', text)       # Italic using _

    # Convert other markdown styles
    text = re.sub(r'\\(\*|\_|\[|\]|\(|\)|\`)', r'\1', text)  # Escape characters
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Images
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Links
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # Headers
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # Unordered lists
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # Ordered lists
    text = re.sub(r'```[\s\S]*?```', '', text)  # Code blocks
    text = re.sub(r'`([^`]*)`', r'\1', text)  # Inline code
    text = re.sub(r'^\s*[-\*]{3,}\s*$', '', text, flags=re.MULTILINE)  # Horizontal rules
    text = re.sub(r'\n+', '\n', text).strip()
    
    return text

# Function to speak text using Google TTS
def speak_text(text):
    try:
        tts = gTTS(text, lang='en', slow=False, tld='com')
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        st.audio(audio_buffer, format="audio/mp3")
    except Exception as e:
        st.error(f"Error in voice output: {e}")

# Function to record and recognize voice input
def record_and_recognize():
    recognizer = sr.Recognizer()
    audio_data = []
    
    with sr.Microphone() as source:
        st.write("Please say something...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        with st.spinner("Listening..."):
            audio = recognizer.listen(source)
            audio_data.append(audio.get_wav_data())

    try:
        if audio_data:
            combined_audio = io.BytesIO()
            for chunk in audio_data:
                combined_audio.write(chunk)
            combined_audio.seek(0)
            
            audio_file = sr.AudioFile(combined_audio)
            with audio_file as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
            return text
        else:
            return ""
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
        return ""
    except sr.RequestError:
        st.error("Could not request results from Google Speech Recognition service.")
        return ""

# Function to correct text using TextBlob
def correct_text(text):
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    return corrected_text

# Main function
def main():
    st.title("YouTube Transcript Q/A Chatbot")

    if "output_placeholder" not in st.session_state:
        st.session_state.output_placeholder = ""

    if "voice_input" not in st.session_state:
        st.session_state.voice_input = ""

    if "recording" not in st.session_state:
        st.session_state.recording = False

    link = st.text_input("Enter YouTube Link")
    st.image(fetch_image(link))

    transcript = ""
    if link:
        try:
            video_id = re.findall(r"v=([A-Za-z0-9_-]+)", link)[0]
            transcript = fetch_transcript(video_id)
            if transcript:
                st.success("Transcript fetched successfully!")
            else:
                st.warning("No transcript found or transcript is disabled for this video.")
        except IndexError:
            st.error("Invalid YouTube link. Please enter a valid YouTube link.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    col1, col2 = st.columns(2)

    with col1:
        # Use the voice input to set the default value of the text area
        question = st.text_area("Question:", value=st.session_state.voice_input or "", height=300, key="question_input")
        
        if st.button("Start Recording"):
            st.session_state.voice_input = record_and_recognize()
            st.rerun()

    with col2:
        output = st.text_area(
            "Answer:",
            value=st.session_state.output_placeholder,
            height=300,
        )

    with col1:
        if st.button("Process"):
            if link and transcript:
                context = PreProcess(transcript, link)
                corrected_question = correct_text(question)
                
                # Check for valid context-question matching
                qa_answer = QNA(corrected_question, context)
                if not qa_answer or qa_answer.lower() == "sorry, no valid answer found.":
                    st.session_state.output_placeholder = "Please ask a valid question related to your video."
                else:
                    extended_prompt = f"""Give answers only and only based on the following question and context, extend the given answer: {qa_answer}\n\nQuestion: {corrected_question}\nContext: {context}\n\nLet me know if you need any kind of help to assist you with this video and if the question is not related to our Context then strictly tell me that, "please enter a valid question related to your video"."""
                    extended_answer = generate_extended_answer(extended_prompt)
                    st.session_state.output_placeholder = markdown_to_text(extended_answer)
                st.rerun()
            else:
                st.warning("No link is entered")

    with col2:
        if st.session_state.output_placeholder:
            speak_text(st.session_state.output_placeholder)

# Streamlit Sidebar for Settings
with st.sidebar:
    st.subheader("Settings")
    st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    st.slider("Max Tokens", min_value=50, max_value=2000, value=1000, step=50)

if __name__ == "__main__":
    main()
