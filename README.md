# 🛍️ **Multimodal Product Retrieval with CLIP and Residual Adapters**

A **CLIP-based multimodal retrieval system** for e-commerce that aligns user text queries with product images in a shared embedding space.  
This project focuses on **domain adaptation without catastrophic misalignment**, achieving significant improvements over zero-shot CLIP while preserving its pre-trained semantic structure.

> 🔍 Example query: *“a small orange plush toy with big eyes”*  
> 🎯 Goal: retrieve visually accurate product images even when metadata is noisy or incomplete.

---

## 📌 **Key Highlights**

- 🚫 Identified **catastrophic misalignment** caused by naive fine-tuning of CLIP embeddings  
- ✅ Proposed a **Residual Projection Head (Adapter)** to preserve pre-trained alignment  
- 📈 Improved **Recall@10 from 58.7% → 69.7%** (+11% absolute)  
- 🧠 Uses **contrastive learning (InfoNCE)** with efficient feature caching  
- ⚙️ Designed with **production-ready retrieval architecture** in mind  
- 🧩 Serves as the **retrieval backbone of a RAG-ready system** (generation layer can be added)

---

## 🧠 **Problem Motivation**

Traditional keyword-based search struggles to bridge the **vocabulary gap** between:

- User intent: *“cute small orange plushie”*  
- Product metadata: *“Toy, Polyester, 5-inch”*

Vision–Language Models (VLMs) such as CLIP offer strong zero-shot alignment, but **naive fine-tuning often degrades performance**, especially on small or noisy datasets.

This project explores a key question:

> **How can we adapt CLIP to an e-commerce domain without destroying its pre-trained semantic structure?**

---

## 🏗️ **System Architecture**

```text
Text Query
   ↓
CLIP Text Encoder
   ↓
Residual Projection Head
   ↓
Text Embedding
   ↓
Vector Database (FAISS / Milvus)
   ↓
ANN Search
   ↓
Top-K Product Images
