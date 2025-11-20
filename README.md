# Dempster–Shafer Uncertainty Quantification for GPT-Style Models

This project implements a Dempster–Shafer–based uncertainty wrapper around a pretrained
LLM (Qwen2.5-3B-GPTQ).  
The goal is to evaluate whether belief functions and ignorance mass can:
- Improve calibration  
- Detect uncertainty  
- Identify hallucinations  
- Enable selective answering  
- Provide interpretable confidence intervals

The model is evaluated on a subset of **MMLU**, using 5 sampling runs per question.

---

## Features

### DST Belief & Ignorance
We use:
- Singleton masses: \( m(\{y_i\}) = \alpha \cdot \bar{p}_i \cdot \rho \)  
- Ignorance mass: \( m(\Theta) = 1 - \sum_i m(\{y_i\}) \)
Where:
- \( \bar{p} \) is the mean softmax across samples  
- \( \rho \) is sampling consistency  
- \( \alpha \) is a normalization chosen based on β  

### Reliability Diagrams (Smoothed)
Uses quantile binning + moving average smoothing.

### Accuracy–Coverage Curves
Shows accuracy when skipping high-ignorance predictions.

### Temperature Scaling Baseline
Fits scalar temperature using calibration split.

### Hallucination Detection
Runs hand-crafted prompts and displays:
- Raw LLM output  
- DST ignorance  
- Belief decomposition  

---

## Installation

```bash
pip install torch datasets transformers scipy matplotlib tqdm numpy pandas
# DST-Project
# DST-Project
