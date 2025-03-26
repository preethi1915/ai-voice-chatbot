import os
import json
import numpy as np
import streamlit as st
import sounddevice as sd
import wave
from dotenv import load_dotenv
from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

# Load API keys
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Initialize OpenAI LLM
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# Initialize OpenAI client
client = OpenAI()

# Load examples for few-shot learning
file_path = "examples.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

examples = [{"input": item["input"], "output": item["output"]} for item in data]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="User: {input}\nBot: {output}"
)

# Semantic Similarity Example Selector
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, OpenAIEmbeddings(), FAISS, k=1
)

similar_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="""You are an AI assistant for Preethi Vallabhadasu, a Python AI Engineer currently working at TCS. 
            Preethi is passionate about AI, Generative AI, and Machine Learning.
            Your task is to answer user queries professionally on behalf od preethi, using the examples provided whenever possible. 

            - If the query is about Preethi and relevant examples are available, use them to craft your response.
            - If no examples are provided, generate a professional response based on general knowledge.
            - If the query is not related to professional aspects of Preethi, politely decline by saying: 
            "I can only answer professional questions about Preethi Vallabhadasu."\n""",
    suffix="Now answer the user's new question.\nUser: {query}\nBot:",
    input_variables=["query"],
)


st.title("🧠 AI Voice BOT")
st.caption("🚀 Your AI Voice Assistant")

if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm your Voicebot. How can I help you today? 🎤"}]

chat_container = st.container()
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

user_query = st.chat_input("Speak or type your question...")

# Function to process text query
def process_query(query):
    formatted_prompt = similar_prompt.format(query=query)
    ai_response = model.invoke([HumanMessage(content=formatted_prompt)])

    # Convert AI response to speech
    audio_response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=ai_response.content
    )
    
    # Save and play audio
    audio_file = "response_audio.mp3"
    with open(audio_file, "wb") as f:
        f.write(audio_response.content)

    os.system(f"start {audio_file}")  # Play audio (Windows), change to 'afplay' for Mac, or 'mpg123' for Linux
    
    return ai_response.content

# Handle user input (text-based)
if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    with st.spinner("🧠 Processing..."):
        ai_response = process_query(user_query)
    
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    st.rerun()

# 🎙 **Real-time Audio Recording & Transcription**
def record_audio(filename="recorded_audio.wav", duration=10, samplerate=44100):
    """ Records audio from the microphone and saves it as a WAV file. """
    print("🎤 Recording... Speak now!")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()  # Wait for recording to finish
    print("✅ Recording finished. Saving...")

    # Save to WAV file
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

    print(f"📁 Audio saved as {filename}")
    return filename

def transcribe_audio(filename="recorded_audio.wav"):
    """ Transcribes audio using OpenAI Whisper. """
    print("📝 Transcribing...")
    with open(filename, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    
    transcribed_text = transcription.text
    print(f"🎙 Transcription:\n{transcribed_text}")
    return transcribed_text

# 🎤 Button to Start Recording
if st.button("🎙 Record Voice Query"):
    audio_file = record_audio()
    transcribed_text = transcribe_audio(audio_file)
    
    st.session_state.message_log.append({"role": "user", "content": transcribed_text})
    
    with st.spinner("🧠 Processing..."):
        ai_response = process_query(transcribed_text)

    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    st.rerun()
