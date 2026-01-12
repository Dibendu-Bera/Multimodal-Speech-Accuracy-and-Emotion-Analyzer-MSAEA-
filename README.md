# Multimodal Speech Accuracy and Emotion Analyzer (MSAEA)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/DibenduBera/Multimodal-Speech-Accuracy-and-Emotion-Analyzer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)](https://github.com/huggingface/peft)

**A modular AI pipeline that simultaneously transcribes speech and detects speaker sentiment, designed to provide granular feedback for intent analysis.**

---

## 🚀 Live Demo
**Try the web application here:** [Hugging Face Spaces Demo](https://huggingface.co/spaces/DibenduBera/Multimodal-Speech-Accuracy-and-Emotion-Analyzer)

---

## 📖 Overview
This project addresses the challenge of analyzing spoken intent by combining **Automatic Speech Recognition (ASR)** with **Sentiment Analysis**. Unlike black-box solutions, this system uses a modular "Ear-Brain-Judge" architecture to process raw audio into meaningful insights.

It was engineered with a focus on resource efficiency, utilizing **QLoRA (Quantized Low-Rank Adaptation)** to fine-tune a Transformer model on consumer-grade hardware while achieving high accuracy.

## 🌟 Key Features
* **Fine-Tuned ASR:** Optimized **OpenAI Whisper (Tiny)** on the LibriSpeech dataset using 8-bit quantization (`bitsandbytes`) and Low-Rank Adapters (`PEFT`).
* [cite_start]**Emotion Detection:** Integrated **DistilRoBERTa** to classify transcribed text into 7 distinct emotions (Joy, Neutral, Anger, etc.) in real-time[cite: 164].
* [cite_start]**Resource Efficiency:** Reduced trainable parameters to just **0.39%** of the base model (approx. 147k params), enabling training on a single T4 GPU[cite: 323].
* [cite_start]**Robust Engineering:** Implemented a custom `DataCollator` to handle multimodal inputs and resolved tensor shape mismatches between audio spectrograms and text labels[cite: 849].
* [cite_start]**Metric-Driven:** Achieved a **Word Error Rate (WER) of 13.9%** on the test set after implementing a custom text normalization pipeline (punctuation stripping & case matching)[cite: 948].

## 🏗️ System Architecture
[cite_start]The project follows a modular pipeline design[cite: 128]:

1.  [cite_start]**The Ear (ASR)**: Converts raw audio waveforms into Log-Mel Spectrograms and predicts text tokens using the fine-tuned Whisper model[cite: 146].
2.  [cite_start]**The Brain (NLP)**: Analyzes the transcribed text using a pre-trained DistilRoBERTa model to detect emotional intent[cite: 159].
3.  [cite_start]**The Judge (Evaluation)**: Compares predictions against reference text using normalized Word Error Rate (WER) metrics[cite: 131].

## 📊 Performance Metrics
| Metric | Result | Description |
| :--- | :--- | :--- |
| **Word Error Rate (WER)** | **13.9%** | [cite_start]Evaluated on the LibriSpeech validation split after normalization[cite: 948]. |
| **Trainable Params** | **0.39%** | Only 147,456 parameters trained; [cite_start]99.6% of weights remained frozen[cite: 323]. |
| **Training Loss** | **3.66** | [cite_start]Final evaluation loss after 3 epochs[cite: 947]. |

## 🛠️ Tech Stack
* **Languages:** Python
* **Deep Learning:** `transformers`, `torch`, `accelerate`
* **Optimization:** `peft` (LoRA), `bitsandbytes` (Quantization)
* **Audio Processing:** `librosa`, `soundfile`
* **Evaluation:** `evaluate`, `jiwer`
* **Deployment:** `gradio`, Hugging Face Spaces

## 📂 Repository Structure

```

├── app.py                 # Main application logic for the Gradio interface
├── requirements.txt       # Dependencies for the deployment environment
├── notebooks/             # (Optional) Jupyter notebooks used for training & experiments
├── README.md              # Project documentation
└── LICENSE                # MIT License

```

## 🚀 Usage (Local)
To run the inference app on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/DibenduBera/MSAEA.git](https://github.com/DibenduBera/MSAEA.git)
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
