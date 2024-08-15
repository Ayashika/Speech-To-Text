# AVCR Approach Using Whisper Model Fine-Tuning and KeyBERT Algorithm

## Introduction

Welcome to our project on improving Automated Voice Command Recognition (AVCR) for cobotic settings. This study introduces a tailored approach utilizing Whisper model fine-tuning and the KeyBERT algorithm to enhance the robustness and accuracy of voice command recognition in environments with prevalent background noise. 

## Project Overview

In cobotic environments, human-robot interactions often rely heavily on voice commands. Traditional Automatic Speech Recognition (ASR) systems may not perform optimally due to the noisy nature of such settings. Our primary motivation for this study is to address these challenges by developing a more robust AVCR system. By fine-tuning the Whisper model and leveraging the KeyBERT algorithm, we aim to significantly improve the accuracy and efficiency of voice command recognition.

### Key Features

1. **Whisper Model Fine-Tuning**: Enhances robustness in handling noise prevalent in cobotic settings.
2. **KeyBERT Algorithm**: Improves semantic understanding of voice commands.
3. **Rigorous Experimentation**: Demonstrates significant AVCR accuracy and robust improvements across diverse cobotic datasets.

## Acoustic Modeling with Whisper

Whisper's design as an encoder-decoder Transformer provides a comprehensive method for speech-to-text conversion. This section outlines the key components and processes involved:

### Audio Segmentation

The audio input is divided into 30-second segments. This segmentation allows for detailed analysis of temporal variations and events in the signal over time.

### Log-Mel Spectrogram Transformation

Each 30-second segment is transformed into a log-Mel spectrogram. This transformation provides a concise representation of the frequency content of the audio signal, which is commonly used as input for tasks like speech recognition or audio classification.

### Temporal Analysis

By dividing the audio into segments, Whisper can analyze temporal variations and capture important events in the audio signal, improving the overall accuracy of the transcription.

## Whisper Model Fine-Tuning

Fine-tuning the Whisper model on our specific dataset is essential for addressing the complexities of cobotic interactions. This process involves:

### Dataset Diversity

To ensure robustness and versatility, our dataset includes a wide range of command structures and contexts. This exposure helps the Whisper model adapt to various scenarios and improve its performance.

### Parameter-Efficient Fine-Tuning

We employ techniques like sparse attention mechanisms and weight pruning to optimize model parameters. These methods reduce the computational requirements and make the fine-tuning process more efficient.

### Low Rank Adaptation (LoRA)

LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture. This approach greatly reduces the number of trainable parameters, making it easier to fine-tune the model for downstream tasks without compromising on performance.

## KeyBERT for Semantic Understanding

KeyBERT enhances the semantic understanding of voice commands by extracting key phrases, which improves command recognition in cobotic tasks. This section details the benefits and applications of KeyBERT in our AVCR system.

### Improved Interaction

KeyBERT enhances the model's ability to understand and respond to specific commands like "start," "stop," or "move". This improvement leads to more effective and efficient human-robot interactions.

### Task-Specific Instructions

In cobotic environments, task-specific instructions are crucial. KeyBERT helps identify commands related to specific tasks, such as picking and placing objects or adjusting settings. This capability enhances the overall workflow efficiency.

### Increased Efficiency

By directly recognizing keywords, the AVCR system can expedite the execution of actions without the need for complex natural language understanding. This leads to faster and more efficient interactions, improving productivity in cobotic environments.

## Evaluation Metrics

Word Error Rate (WER) is the primary metric used to evaluate the performance of our ASR system. WER is particularly suitable for tasks where the order and alignment of generated elements (words or tokens) are crucial. Our model achieved a WER of 0.29 over 600 evaluation steps (3 epochs), indicating high accuracy in command recognition.

## Implementation Details

### Model Architecture

- **Encoder-Decoder Transformer**: The core architecture of the Whisper model.
- **Log-Mel Spectrograms**: Used as input for robust speech recognition.

### Training Methodology

1. **Pre-processing**: Preparing the dataset by cleaning and formatting the audio and text data.
2. **Model Selection**: Choosing the appropriate architecture and parameters for the Whisper model.
3. **Hyperparameter Optimization**: Fine-tuning hyperparameters to achieve optimal performance.

### Parameter-Efficient Techniques

- **Sparse Attention Mechanisms**: Reduce the computational requirements for the model.
- **Weight Pruning**: Optimize model parameters by removing redundant weights.
- **LoRA (Low Rank Adaptation)**: Enhance training throughput with fewer parameters.

## Command Recognition and Keyword Classification

In a cobotic environment, command recognition and keyword classification are crucial for efficient human-robot interaction. This section explores how our AVCR system addresses these aspects.

### Command Recognition

Humans often give commands to robots through speech. Keyword classification helps in identifying specific commands or keywords that trigger certain actions. For example, keywords like "start," "stop," or "move" can be recognized to control the robot's behavior.

### Task-Specific Instructions

Keyword classification allows the ASR system to identify task-specific instructions or requests. This can include commands related to picking and placing objects, adjusting settings, or requesting information.

### Increased Efficiency

By directly recognizing keywords, the ASR system can expedite the execution of actions without the need for complex natural language understanding. This can lead to faster and more efficient human-robot interactions, improving overall workflow in the cobotic environment.

## Conclusion

The deployment of the Whisper model in speech-to-text recognition has yielded compelling outcomes, showcasing its prowess in converting spoken language into written form. Through meticulous training methodologies, encompassing pre-processing, model architecture selection, and hyperparameter optimization, we've witnessed substantial enhancements in the precision and performance of our speech-to-text models. Our tailored approach, combining Whisper model fine-tuning with KeyBERT, addresses the unique challenges of cobotic environments and significantly improves AVCR accuracy.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA
- Whisper Model
- KeyBERT


## Acknowledgments

This README provides a comprehensive overview of our project, its features, implementation details, and usage instructions.
