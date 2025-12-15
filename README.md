# Exploring KL Regularization in Reinforcement Learning for LLMs

This repository contains the code and experiments for an empirical study on **KL divergence–based regularization** in reinforcement learning (RL) fine-tuning of large language models (LLMs).

We compare **reverse KL**, **forward KL**, and **no KL regularization** in modern RLHF-style training setups, and analyze their effects on:
- Policy entropy
- Output diversity
- Training accuracy
- Exploration behavior

Our results challenge common intuitions about KL direction (mode-seeking vs. mode-covering) when applied to LLM policy optimization.

---

## 🔍 Overview

KL regularization is widely used in RL fine-tuning of LLMs to prevent policy drift from a reference model. While reverse KL is often described as *mode-seeking* and forward KL as *mode-covering*, their practical impact on exploration and diversity during LLM RL training is not well understood.

This project investigates whether the **direction of KL regularization alone** meaningfully affects:
- Diversity of generated reasoning traces
- Entropy collapse during RL training
- Performance on reasoning-heavy tasks

---

## 🧪 Experimental Setup

- **Model:** DeepSeek-R1-Distill-Qwen-1.5B  
- **RL Algorithm:** PPO with RLOO (Leave-One-Out) advantage estimator  
- **Frameworks:**  
  - [`verl`](https://github.com/volcengine/verl) for RL training  
  - `vLLM` for high-throughput sampling  

### Tasks / Datasets
- **DeepScaleR**: Math reasoning problems (Agentica)
- **Palindrome**: Procedural generation task from Reasoning Gym with multiple valid outputs

Each experiment compares:
1. **No KL regularization**
2. **Reverse KL** (K1 estimator added to the reward)
3. **Forward KL-style** regularization (K3 estimator added to the loss)

---

## 📊 Evaluation Metrics

We analyze both performance and diversity using:
- **Training accuracy**
- **Token-level policy entropy**
- **Cosine similarity** between response embeddings
- **Response uniqueness** (number of distinct valid outputs, for Palindrome)

---

## 📌 Key Findings

- **No consistent diversity differences** were observed between forward KL, reverse KL, and no KL.
- **Entropy metrics** did not clearly separate KL directions.
- **No-KL training** achieved the highest training accuracy across tasks.
- Common intuitions about KL direction from variational inference **do not directly translate** to LLM RL fine-tuning.

These results suggest that **KL estimator choice and interaction with policy gradients** may matter more than KL direction alone.

---

## ⚠️ Limitations & Future Work

- Experiments were conducted on a relatively small model (1.5B parameters).
- Training horizons were limited.
- Future work could explore:
  - Larger models and longer training runs
  - Explicit diversity objectives
  - Alternative divergence measures or hybrid KL strategies

---

## 👥 Team

- Leonardo Falvo  
- Élodie Côté-Gauthier  
- Aditi Potnis  
- Vedant Shah  
- Vineet Jain  

Affiliations: McGill University, Université de Montréal, Mila – Quebec AI Institute
