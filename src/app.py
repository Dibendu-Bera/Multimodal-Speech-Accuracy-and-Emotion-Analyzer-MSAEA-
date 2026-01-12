import gradio as gr
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from peft import PeftModel
import torch
import os

# --- CONFIGURATION ---
# REPLACE THIS WITH YOUR USERNAME
hf_username = "DibenduBera" 
peft_model_id = f"{hf_username}/whisper-tiny-msaea-finetuned"

# --- LOAD MODELS ---
print(f"🔄 Downloading model: {peft_model_id}")

try:
    # 1. Load Base Model (Explicitly for CPU)
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-tiny",
        device_map="cpu",  # Force CPU
        low_cpu_mem_usage=True
    )
    
    # 2. Load Adapters
    # We ignore the 8-bit config from training to make it run on CPU
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    
    # 3. Load Processor
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")

    # 4. Create ASR Pipeline
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30
    )
    
    # 5. Load Emotion Pipeline
    print("❤️ Loading emotion analyzer...")
    emo_pipe = pipeline(
        'text-classification', 
        model='j-hartmann/emotion-english-distilroberta-base'
    )

except Exception as e:
    print(f"❌ CRITICAL ERROR LOADING MODELS: {e}")
    raise e

# --- PROCESSING FUNCTION ---
def analyze_intent(audio_filepath):
    if not audio_filepath:
        return "No audio provided", "N/A"
    
    try:
        # Step 1: Transcribe
        print("🎤 Processing audio...")
        transcription = asr_pipe(audio_filepath)["text"]
        
        # Step 2: Analyze Emotion
        print("🧠 Analyzing emotion...")
        emotion_result = emo_pipe(transcription)
        top_emotion = emotion_result[0]['label']
        score = emotion_result[0]['score']
        
        emotion_text = f"{top_emotion.upper()} ({score:.1%})"
        
        return transcription, emotion_text
        
    except Exception as e:
        return f"Error: {str(e)}", "Error"

# --- INTERFACE ---
interface = gr.Interface(
    fn=analyze_intent,
    inputs=gr.Audio(type="filepath", label="Speak or Upload Audio"),
    outputs=[
        gr.Textbox(label="Transcription (The Ear)"),
        gr.Label(label="Detected Tone (The Heart)")
    ],
    title="Multimodal Speech Accuracy and Emotion Analyzer (MSAEA)",
    description="This app uses a fine-tuned Whisper model to transcribe speech and a DistilRoBERTa model to detect emotion."
)

if __name__ == "__main__":
    interface.launch()
