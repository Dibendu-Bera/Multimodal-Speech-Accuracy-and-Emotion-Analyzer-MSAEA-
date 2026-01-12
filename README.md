Here is the clean Markdown code for your `README.md` with all citation markers removed.

```markdown
# Multimodal Speech Accuracy and Emotion Analyzer (MSAEA)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/DibenduBera/Multimodal-Speech-Accuracy-and-Emotion-Analyzer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)](https://github.com/huggingface/peft)
[![Whisper](https://img.shields.io/badge/Model-Whisper%20Tiny-green)](https://huggingface.co/openai/whisper-tiny)

**A modular AI pipeline that simultaneously transcribes speech and detects speaker sentiment, designed to provide granular feedback for intent analysis.**

---

## 🚀 Live Demo
**Try the web application here:** [Hugging Face Spaces Demo](https://huggingface.co/spaces/DibenduBera/Multimodal-Speech-Accuracy-and-Emotion-Analyzer)

---

## 📖 Overview
This project addresses the challenge of analyzing spoken intent by combining **Automatic Speech Recognition (ASR)** with **Sentiment Analysis**. Unlike black-box solutions, this system uses a modular pipeline architecture to process raw audio into meaningful insights.

It was engineered with a focus on resource efficiency, utilizing **QLoRA (Quantized Low-Rank Adaptation)** to fine-tune a Transformer model on consumer-grade hardware while achieving high accuracy.

## 🏗️ System Architecture
The system follows a 3-block "Blueprint" design for modular control:

1.  **Block 1: The Ear (ASR)** 👂
    * **Model:** OpenAI Whisper (Tiny).
    * **Role:** Converts raw audio waveforms into text transcripts. Optimized with 8-bit quantization for memory efficiency.
2.  **Block 2: The Brain (NLP)** 🧠
    * **Model:** DistilRoBERTa (`j-hartmann/emotion-english-distilroberta-base`).
    * **Role:** Analyzes the transcribed text to detect 7 distinct emotional intents (Joy, Neutral, Anger, etc.).
3.  **Block 3: The Judge (Evaluation)** ⚖️
    * **Metric:** Word Error Rate (WER).
    * **Role:** Objectively scores pronunciation accuracy by comparing predictions against reference text.

## 🌟 Key Features
* **Fine-Tuned Accuracy:** Achieved a **Word Error Rate (WER) of 13.9%** on the test set after implementing a custom text normalization pipeline (punctuation stripping & case matching).
* **Efficient Training:** Reduced trainable parameters to just **0.39%** (147k params) of the base model using LoRA Rank=8, allowing training on a single T4 GPU.
* **Multimodal Integration:** Seamlessly chains audio processing and text classification into a single inference pipeline.
* **Robust Engineering:** Implemented a custom `DataCollator` to handle multimodal inputs and resolved tensor shape mismatches between audio spectrograms and text labels.

## 📊 Performance Metrics
| Metric | Result | Context |
| :--- | :--- | :--- |
| **Word Error Rate (WER)** | **13.94%** | Evaluated on LibriSpeech validation split. |
| **Trainable Params** | **0.39%** | 147,456 params trained vs 37M frozen. |
| **Training Loss** | **3.66** | Final evaluation loss after 3 epochs. |

## 🛠️ Tech Stack
* **Deep Learning:** `transformers`, `torch`, `accelerate`
* **Optimization:** `peft` (LoRA), `bitsandbytes` (Quantization)
* **Audio Processing:** `librosa`, `soundfile`
* **Evaluation:** `evaluate`, `jiwer`
* **Deployment:** `gradio`, Hugging Face Spaces

## 📂 Repository Structure

```

├── app.py                 # Main application logic for the Gradio interface
├── requirements.txt       # Dependencies for the deployment environment
├── notebooks/             # Training notebook with QLoRA implementation
├── README.md              # Project documentation
└── LICENSE                # MIT License

```

## 🚀 Usage (Local)
To run the inference app on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [[https://github.com/Dibendu-Bera/MSAEA](https://github.com/Dibendu-Bera/Multimodal-Speech-Accuracy-and-Emotion-Analyzer-MSAEA-).git]([https://github.com/DibenduBera/MSAEA](https://github.com/Dibendu-Bera/Multimodal-Speech-Accuracy-and-Emotion-Analyzer-MSAEA-).git)
    cd MSAEA
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python app.py
    ```

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
