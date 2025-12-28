# 🛍️ Multimodal Product Retrieval with CLIP and Residual Adapters

A CLIP-based multimodal retrieval system for e-commerce that aligns user text queries with product images in a shared embedding space.

This project focuses on **domain adaptation without catastrophic misalignment**, achieving significant improvements over zero-shot CLIP while preserving its pre-trained semantic structure.

---

## 🔍 Example Query

> **“a small orange plush toy with big eyes”**

🎯 **Goal**  
Retrieve visually accurate product images even when metadata is noisy, incomplete, or weakly descriptive.

---

## 📌 Key Highlights

- 🚫 Identified catastrophic misalignment caused by naive fine-tuning of CLIP embeddings
- 🔄 Iterative experimental process: naive MLP → CLIP baseline → residual adapters
- ✅ Proposed a **Residual Projection Head (Adapter)** to preserve pre-trained alignment
- 🧠 Uses **contrastive learning (InfoNCE)** with learnable temperature
- 📈 Improved **Recall@10 from 58.7% → 69.7% (+11% absolute)**
- ⚙️ Uses efficient feature caching for scalable experimentation
- 🧩 Designed as the retrieval backbone of a RAG-ready system (generation layer can be added)

---

## 🧠 Problem Motivation

Traditional keyword-based search struggles to bridge the vocabulary gap between:

- **User intent**  
  “cute small orange plushie”

- **Product metadata**  
  “Toy, Polyester, 5-inch”

Vision–Language Models (VLMs) such as CLIP offer strong zero-shot alignment between text and images.  
However, **naive fine-tuning often degrades performance**, especially on small or noisy datasets.

This project explores the following question:

> **How can we adapt CLIP to an e-commerce domain without destroying its pre-trained semantic structure?**

---

## 🧪 Experimental Process

### Step 1: Naive Two-Tower MLP (Initial Attempt)

- Trained a simple two-tower MLP on top of CLIP embeddings
- Used margin-based contrastive loss with random negative sampling

**Result**  
Performance was **worse than direct CLIP cosine similarity**, indicating catastrophic misalignment and loss of semantic structure.

---

### Step 2: Zero-Shot CLIP Baseline

- Used frozen CLIP image and text embeddings
- Retrieval via cosine similarity

**Baseline Performance**
- Recall@10 ≈ **58.7%**

---

### Step 3: Residual Projection Head + InfoNCE (Final Model)

To improve performance without destroying CLIP semantics, we introduced:

- Lightweight **Residual Projection Heads** on image and text embeddings
- Frozen CLIP backbone (only heads are trained)
- **InfoNCE contrastive loss** using batch-wise negatives
- **Learnable temperature (logit scale)**

This approach allows **small, controlled adaptation** of CLIP embeddings to the target domain.

**Final Performance**
- Recall@10 ≈ **69.7%**
- Consistent improvements across Recall@1 / Recall@5 / MRR
```

---

## 🏗️ System Architecture

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
(Image embeddings follow the same pipeline symmetrically.)

```

---

## 📊 Evaluation

**Metrics**
- Recall@1
- Recall@5
- Recall@10
- MRR

**Models Compared**
- Zero-shot CLIP
- Naive two-tower MLP
- Residual Adapter + InfoNCE (ours)

**Analysis**
- Quantitative retrieval metrics
- Qualitative visualization of retrieved images
- Detailed error analysis across different query types

---

## 🧠 Assistant / TA Comments (Original)

**Comments**  
This project develops a multimodal text-to-image retrieval system for e-commerce.  
Building on the pretrained CLIP model, the work follows an iterative experimental process:  
starting with a naive two-tower MLP, comparing performance with a zero-shot CLIP baseline,  
and ultimately designing a residual projection head trained with InfoNCE loss.

You experimented with fine-tuning, temperature parameters, and InfoNCE loss, demonstrating strong effort to improve model performance despite the challenges.

**Implementation**  
The report would benefit from more detail on the fine-tuning process, hyperparameter-tuning strategy, and the rationale behind specific choices (e.g., why alpha = 0.2 in the residual projection head).

**Results and Evaluation**  
Excellent work on the detailed error analysis across models.

**Business Case & Deployment**  
The report could be strengthened with a more thorough discussion of the business case, development plan, and potential deployment considerations.

---

## 🧠 Project Reflection (Response to Feedback)

**What Worked Well**
- Identified and diagnosed catastrophic misalignment early
- Systematic comparison between naive fine-tuning and CLIP baseline
- Effective use of residual adapters to preserve semantic alignment
- Strong quantitative and qualitative evaluation

**Limitations**
- Hyperparameter choices (e.g., residual scaling factor α) were not fully ablated
- Fine-tuning strategy could be explored more systematically
- Business case and deployment considerations were not deeply analyzed

---

## 🚀 Future Work

- Partial unfreezing of CLIP backbone (Stage B fine-tuning)
- Hard negative mining for contrastive learning
- Larger-scale deployment with FAISS or Milvus
- Integration with a generative layer for multimodal RAG systems
- Latency benchmarking and production indexing strategy

---

## 📌 Summary

This project demonstrates that effective domain adaptation of CLIP does not require full retraining.

Lightweight residual adapters combined with contrastive learning can significantly improve retrieval performance while preserving CLIP’s pretrained semantic structure.
