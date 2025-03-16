# **AttentionExperts: Mixture of Attention Experts (MoAE)**  

**AttentionExperts** is a simple yet effective implementation of the **Mixture of Attention Experts (MoAE)**, an extension of the Mixture of Experts (MoE) framework where individual experts specialize in attention mechanisms rather than traditional feedforward layers.

## **Overview**  
This repository provides a **distributed and scalable implementation** of MoAE, allowing for efficient **parallelized training** of attention-based expert models. The base MoE implementation is adapted from [st-moe](https://github.com/lucidrains/st-moe-pytorch), a high-performance parallelized MoE framework.

## Install requirements
```sh
pip install -r requirements.txt
```

## **Code Structure**  
-  **`MoAE_train.py`** – Complete training script for a simple **MoAE Transformer** model.  
- **`testing_Expert.py`** – Unit tests for standard **feedforward experts** to validate outputs and gradients.  
- **`testing_ExpertAttention.py`** – Unit tests for **attention-based experts** to ensure correct functionality.  

### Training
The training example is same as Karapathy's NanoGPT training code trained on Shakespear text as a causal model for predicting next letter. To train run, 
```sh
python MoAE_train.py
``` 


