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
* **Emotion Detection:** Integrated **DistilRoBERTa** to classify transcribed text into 7 distinct emotions (Joy, Neutral, Anger, etc.) in real-time.
* **Resource Efficiency:** Reduced trainable parameters to just
