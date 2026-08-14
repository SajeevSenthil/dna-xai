# Explainable DNA Sequence Classification using DNABERT-2

## 1. Project Title

**Explainable Promoter and Transcription Factor Binding Site Prediction using DNABERT-2 and Parameter-Efficient Fine-Tuning**

### Repository Name

`dna-xai`

---

# 2. Project Overview

This project aims to develop an **explainable deep learning system for DNA sequence analysis**, specifically for:

1. **Promoter region prediction**
2. **Transcription Factor Binding Site (TFBS) prediction**

The project will use **DNABERT-2** as the primary DNA foundation model.

Instead of simply fine-tuning DNABERT-2 using one method, the project will perform a comparative study of different parameter-efficient fine-tuning techniques:

- LoRA
- QLoRA
- Adapters

A conventional **full fine-tuning model** will be used as the baseline.

The final system will not only predict whether a DNA sequence contains a promoter/TFBS but will also provide an **interpretation of which parts of the DNA sequence contributed to the prediction**.

SHAP/LIME-based attribution will be used to identify important sequence regions, which can then be converted into a human-readable biological explanation.

---

# 3. Core Research Question

The central question of the project is:

> **Can DNABERT-2 efficiently predict regulatory DNA elements using parameter-efficient fine-tuning while providing interpretable evidence for its predictions?**

The project therefore has two major components:

### Prediction

```text
DNA Sequence
      ↓
DNABERT-2
      ↓
Fine-tuned representation
      ↓
Classification Head
      ↓
Promoter / TFBS Prediction
```

### Explanation

```text
DNA Sequence
      ↓
Model Prediction
      ↓
SHAP / LIME
      ↓
Important nucleotides / k-mers / regions
      ↓
Biological motif analysis
      ↓
Human-readable explanation
```

---

# 4. Biological Motivation

DNA contains regulatory regions that control gene expression.

A **promoter** is a regulatory DNA region involved in initiating transcription.

A **transcription factor binding site** is a short DNA sequence to which a transcription factor can bind and influence gene regulation.

These regions can contain characteristic sequence patterns or motifs.

For example:

```text
...CGCGCG...TATAAA...GCGCG...
             ↑
         possible motif
```

A deep learning model can learn patterns that are difficult to manually identify.

However, a major problem is:

> The model may correctly predict a promoter without telling us why.

Therefore, interpretability is an important part of this project.

Instead of reporting only:

```text
Prediction: Promoter
Probability: 0.96
```

the system should ideally produce:

```text
Prediction: Promoter
Confidence: 96%

Important region:
Positions 27–34

Detected pattern:
TATAAA

Interpretation:
The model assigned high importance to the region around
positions 27–34, which contains a TATA-like motif commonly
associated with promoter regions.
```

---

# 5. Overall Architecture

```text
                    DNA Sequence
                         │
                         ▼
                Sequence Preprocessing
                         │
                         ▼
                    DNABERT-2
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Fine-Tuning Method      Baseline
              │              Full Fine-Tuning
              │
     ┌────────┼────────┐
     │        │        │
     ▼        ▼        ▼
   LoRA     QLoRA   Adapters
     │        │        │
     └────────┼────────┘
              │
              ▼
       Classification Head
              │
              ▼
      Prediction + Probability
              │
              ▼
       Explainability Layer
              │
        ┌─────┴─────┐
        ▼           ▼
      SHAP         LIME
        │           │
        └─────┬─────┘
              ▼
     Important DNA Regions
              │
              ▼
       Motif Identification
              │
              ▼
    Biological Interpretation
              │
              ▼
      Natural-Language Output
```

---

# 6. Task Definition

The project can be implemented as two related binary classification tasks.

## Task A — Promoter Prediction

Input:

```text
DNA sequence
```

Output:

```text
Promoter
Non-Promoter
```

Example:

```text
Input:
ATGCGTACGTTAGCGT...

Output:
Promoter: 0.94
Non-Promoter: 0.06
```

---

## Task B — TFBS Prediction

Input:

```text
DNA sequence
```

Output:

```text
TFBS
Non-TFBS
```

Example:

```text
Input:
CGTACGTAGCTAGCTA...

Output:
TFBS: 0.89
Non-TFBS: 0.11
```

The two tasks can initially be implemented independently.

If time permits, they can later be combined into a multi-task model.

---

# 7. Model Selection

## DNABERT-2

DNABERT-2 will be the primary sequence model.

It is appropriate because it is specifically designed for genomic sequence representation learning rather than using a general NLP transformer.

The model will convert DNA sequences into contextual representations.

Conceptually:

```text
DNA

A C G T A C G T A ...

        ↓

Tokenizer

        ↓

DNA Tokens

        ↓

DNABERT-2

        ↓

Contextual Embeddings

        ↓

Classification Head
```

---

# 8. Baseline

Before applying parameter-efficient fine-tuning, establish a baseline.

## Full Fine-Tuning

All or most DNABERT-2 parameters are updated during training.

```text
DNABERT-2
   │
   ├── Parameters updated
   │
   ▼
Classification Head
```

Record:

- Accuracy
- Precision
- Recall
- F1-score
- AUROC
- Training time
- GPU memory
- Number of trainable parameters

This provides the reference against which LoRA, QLoRA and Adapters can be compared.

---

# 9. Fine-Tuning Method 1 — LoRA

LoRA stands for **Low-Rank Adaptation**.

Instead of updating the original model weights, small trainable low-rank matrices are added to selected layers.

Conceptually:

```text
Original DNABERT-2
       │
       ├── Frozen weights
       │
       └── Small LoRA matrices
                  │
                  ▼
             Task adaptation
```

Advantages:

- Very few trainable parameters
- Lower memory consumption
- Faster training
- Original model remains mostly frozen

Metrics to record:

```text
Trainable Parameters
GPU Memory
Training Time
Validation F1
Test F1
```

---

# 10. Fine-Tuning Method 2 — QLoRA

QLoRA combines:

- Quantization
- LoRA

The base model is quantized to reduce memory requirements while LoRA adapters remain trainable.

Conceptually:

```text
DNABERT-2
    │
    ▼
Quantized Model
    │
    ▼
LoRA Adapters
    │
    ▼
Task-specific prediction
```

The main question is:

> How much performance can be retained while significantly reducing memory requirements?

This makes QLoRA particularly useful when working with limited GPU resources.

---

# 11. Fine-Tuning Method 3 — Adapters

Adapters introduce small trainable neural modules inside the transformer.

```text
Transformer Layer

Attention
    │
    ▼
Adapter
    │
    ▼
Feed Forward
    │
    ▼
Next Layer
```

The original DNABERT-2 parameters remain mostly frozen.

This gives another parameter-efficient strategy that is structurally different from LoRA.

---

# 12. Final Fine-Tuning Comparison

The main experiment will therefore contain:

| Model | Purpose |
|---|---|
| Full Fine-Tuning | Baseline |
| LoRA | PEFT comparison |
| QLoRA | Memory-efficient PEFT |
| Adapters | Alternative PEFT architecture |

Every method must use:

- Same dataset
- Same train/validation/test split
- Same preprocessing
- Same classification task
- Same evaluation metrics

This ensures that the comparison is fair.

---

# 13. Classification Head

The DNABERT-2 representation will be passed to a classification head.

Example:

```text
DNABERT-2
    │
    ▼
Sequence Representation
    │
    ▼
Dropout
    │
    ▼
Dense Layer
    │
    ▼
Classification Layer
    │
    ▼
Sigmoid
```

For binary classification:

```text
Output ∈ [0, 1]
```

Example:

```text
0.96 → Promoter
0.04 → Non-Promoter
```

---

# 14. Explainability Component

This is the important additional component of the project.

The model should answer two questions:

### Question 1

> What did the model predict?

Example:

```text
Promoter = 96%
```

### Question 2

> Which parts of the DNA sequence caused the prediction?

Example:

```text
Positions 24–31
TATAAA
High importance
```

This second component is handled using explainability methods.

---

# 15. SHAP

SHAP can be used to estimate the contribution of input features toward the model prediction.

For DNA sequences, the features can be represented as:

- nucleotides
- k-mers
- tokenized DNA segments
- model embeddings

The explanation can look conceptually like:

```text
Sequence:

ATGCGTATAAAGCGTACG...

          ↑↑↑↑↑↑
          Important
```

The important positions can be ranked by their SHAP contribution.

Example:

```text
Position    Base    SHAP Contribution

24          T       +0.21
25          A       +0.18
26          T       +0.17
27          A       +0.15
28          A       +0.13
29          A       +0.12
```

These values can then be visualized or converted into textual explanations.

---

# 16. LIME

LIME provides another way of identifying important parts of an input.

For DNA sequences, the input can be divided into local regions or k-mers.

Example:

```text
Original:

ATGCGTATAAAGCGTACGTAGC
      └──────┘
       region
```

LIME can perturb different regions and observe how the prediction changes.

For example:

```text
Remove region A → prediction drops from 0.94 to 0.61
Remove region B → prediction drops from 0.94 to 0.91
Remove region C → prediction drops from 0.94 to 0.89
```

Therefore:

```text
Region A
    ↓
High contribution
```

---

# 17. SHAP vs LIME

The project does not need to use both as equally important methods.

A practical approach is:

### Primary method

**SHAP**

### Secondary validation method

**LIME**

Then compare whether both methods identify similar important DNA regions.

Example:

```text
                SHAP       LIME

Region 1        High       High
Region 2        Low        Low
Region 3        High       Medium
Region 4        Low        Low
```

If both methods consistently identify the same region, the explanation becomes more convincing.

---

# 18. From Numerical Explanation to Text

This is where the project becomes more interesting.

Instead of stopping at:

```text
SHAP value = +0.21
```

convert the attribution into a biological explanation.

Example:

```text
Prediction:
Promoter

Confidence:
96%

Important region:
Positions 24–31

Sequence:
TATAAA

SHAP contribution:
High positive contribution

Explanation:

"The model strongly relied on the region around positions
24–31. This region contains a TATA-like sequence motif,
which is commonly associated with promoter architecture."
```

This can initially be generated using a deterministic template.

This is preferable to immediately introducing another large language model because the explanation remains **grounded in measurable model evidence**.

---

# 19. Biological Motif Analysis

The attribution method tells us:

> Which region did the model consider important?

The motif analysis tells us:

> Does that region have biological meaning?

The workflow becomes:

```text
DNA Sequence
      │
      ▼
DNABERT-2 Prediction
      │
      ▼
SHAP / LIME
      │
      ▼
Important Region
      │
      ▼
Motif Search
      │
      ▼
Known Regulatory Motif
      │
      ▼
Explanation
```

Potential motif resources include databases such as:

- JASPAR
- HOCOMOCO

The project should use one primary motif database to keep the implementation manageable.

---

# 20. Example Final Output

The final system could produce something like:

```text
==================================================
DNA REGULATORY ELEMENT PREDICTION
==================================================

Prediction:
Promoter

Confidence:
96.4%

--------------------------------------------------
IMPORTANT SEQUENCE REGION
--------------------------------------------------

Position:
27–34

Sequence:
TATAAA

SHAP contribution:
+0.31

LIME importance:
High

--------------------------------------------------
BIOLOGICAL INTERPRETATION
--------------------------------------------------

The model strongly relied on the region around
positions 27–34.

This region contains a TATA-like motif, which is
associated with promoter regions and transcription
initiation.

The agreement between SHAP and LIME indicates that
this region consistently contributes to the model's
promoter prediction.

--------------------------------------------------
FINAL INTERPRETATION
--------------------------------------------------

The sequence is predicted to be a promoter with
high confidence, with the TATA-like region being
one of the strongest contributing sequence patterns.
==================================================
```

This is the type of output that makes the project genuinely **explainable**, rather than simply producing a prediction.

---

# 21. Experimental Design

Every model will be evaluated under the same conditions.

## Experiment 1

```text
DNABERT-2
+
Full Fine-Tuning
```

Purpose:

Establish baseline performance.

---

## Experiment 2

```text
DNABERT-2
+
LoRA
```

Purpose:

Measure the performance/efficiency tradeoff.

---

## Experiment 3

```text
DNABERT-2
+
QLoRA
```

Purpose:

Evaluate quantized PEFT.

---

## Experiment 4

```text
DNABERT-2
+
Adapters
```

Purpose:

Compare an alternative PEFT architecture.

---

# 22. Evaluation Metrics

Do not rely only on accuracy.

The following metrics should be reported:

### Classification

- Accuracy
- Precision
- Recall
- F1-score
- AUROC
- Confusion Matrix

### Efficiency

- Total parameters
- Trainable parameters
- Percentage of trainable parameters
- GPU memory
- Training time
- Inference time

### Explainability

- Number of important regions identified
- SHAP/LIME agreement
- Motif overlap
- Biological validity
- Explanation consistency

---

# 23. Main Comparison Table

The final report should contain a table like:

| Method | Accuracy | Precision | Recall | F1 | AUROC | Trainable Params | GPU Memory | Training Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full FT | | | | | | | | |
| LoRA | | | | | | | | |
| QLoRA | | | | | | | | |
| Adapters | | | | | | | | |

Then a separate explainability table:

| Method | Important Region | SHAP/LIME Agreement | Motif Found | Biological Interpretation |
|---|---|---|---|---|
| Full FT | | | | |
| LoRA | | | | |
| QLoRA | | | | |
| Adapters | | | | |

---

# 24. Research Questions

The project can be structured around four research questions.

### RQ1

> How effectively can DNABERT-2 identify promoter and TFBS regions?

### RQ2

> How do LoRA, QLoRA and Adapters compare with full fine-tuning?

### RQ3

> Can parameter-efficient fine-tuning achieve comparable predictive performance with substantially fewer trainable parameters?

### RQ4

> Can SHAP/LIME identify sequence regions that correspond to biologically meaningful regulatory motifs?

These questions give the project a proper experimental/research structure.

---

# 25. Hypotheses

### H1

DNABERT-2 will perform well on promoter/TFBS classification because it is pretrained specifically on genomic sequences.

### H2

LoRA, QLoRA and Adapters will achieve competitive performance while training significantly fewer parameters than full fine-tuning.

### H3

QLoRA will substantially reduce GPU memory requirements.

### H4

Important regions identified by SHAP/LIME will overlap with known biological motifs more often than randomly selected regions.

---

# 26. Dataset Pipeline

The preprocessing pipeline should be:

```text
Raw Dataset
     │
     ▼
Remove Invalid Sequences
     │
     ▼
Normalize DNA Characters
     │
     ▼
Remove Duplicates
     │
     ▼
Create Labels
     │
     ▼
Train / Validation / Test Split
     │
     ▼
DNABERT-2 Tokenization
     │
     ▼
Model Training
```

Important:

The train/test split should avoid leakage.

If highly similar DNA sequences appear in both training and test sets, the model may appear to perform much better than it actually does.

---

# 27. Project Repository Structure

```text
dna-xai/
│
├── README.md
├── plan.md
├── requirements.txt
├── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_lora.ipynb
│   ├── 04_qlora.ipynb
│   ├── 05_adapters.ipynb
│   └── 06_explainability.ipynb
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   └── dataset.py
│   │
│   ├── models/
│   │   ├── dnabert.py
│   │   ├── classifier.py
│   │   └── peft_models.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   ├── explainability/
│   │   ├── shap_explainer.py
│   │   ├── lime_explainer.py
│   │   └── explanation.py
│   │
│   ├── motifs/
│   │   └── motif_analysis.py
│   │
│   └── utils/
│       └── metrics.py
│
├── experiments/
│   ├── full_finetuning/
│   ├── lora/
│   ├── qlora/
│   └── adapters/
│
├── results/
│   ├── metrics/
│   ├── plots/
│   └── explanations/
│
└── reports/
    ├── figures/
    └── tables/
```

---

# 28. Development Phases

## Phase 1 — Literature and Problem Understanding

Study:

- Promoters
- TFBS
- DNA sequence representation
- Transformer architecture
- DNABERT-2
- PEFT
- Explainable AI
- SHAP
- LIME

Deliverable:

```text
Literature review
Problem definition
Dataset selection
```

---

## Phase 2 — Dataset

Implement:

```text
Download
    ↓
Clean
    ↓
Analyze
    ↓
Split
    ↓
Tokenize
```

Deliverable:

Clean reproducible dataset pipeline.

---

## Phase 3 — Baseline

Implement full fine-tuning.

Deliverable:

```text
Baseline metrics
Baseline model
Training logs
```

---

## Phase 4 — PEFT Experiments

Implement:

```text
LoRA
QLoRA
Adapters
```

Deliverable:

Four comparable experimental results:

```text
Full FT
LoRA
QLoRA
Adapters
```

---

## Phase 5 — Explainability

Implement:

```text
SHAP
LIME
```

Test them on correctly and incorrectly classified examples.

Deliverable:

```text
Important DNA regions
Attribution scores
Visualizations
```

---

## Phase 6 — Biological Interpretation

Connect important regions with known motifs.

```text
Attribution
    ↓
Important sequence
    ↓
Motif search
    ↓
Biological interpretation
```

Deliverable:

Human-readable explanations.

---

## Phase 7 — Comparative Analysis

Compare:

```text
Performance
+
Parameter Efficiency
+
Memory
+
Training Time
+
Interpretability
```

Deliverable:

Final comparison tables and graphs.

---

# 29. Final System

The final system should accept:

```text
DNA sequence
```

and return:

```text
Prediction
Confidence
Important sequence region
SHAP/LIME contribution
Detected motif
Biological interpretation
```

Example:

```text
Input:
>sequence_001
ATCGCGTATAAAGCTAGCG...

Prediction:
Promoter

Confidence:
96.4%

Important region:
25–32

Important sequence:
TATAAA

Explanation:
The model primarily relied on the region around
positions 25–32. This region contains a TATA-like
motif associated with promoter activity. Both the
model attribution and motif analysis support the
promoter classification.
```

---

# 30. Expected Contribution

The project is not claiming to create a new DNA foundation model.

Instead, its contribution is the **systematic comparison and interpretability pipeline**:

```text
DNABERT-2
    +
PEFT comparison
    +
Regulatory DNA prediction
    +
SHAP/LIME
    +
Biological motif grounding
    =
Explainable DNA sequence classification
```

The strongest part of the project is therefore the combination of:

**Performance + Parameter Efficiency + Interpretability + Biological Evidence.**

---

# 31. What NOT to Overcomplicate

For the course project, avoid initially adding:

- Another large language model
- Complex multi-agent systems
- A custom generative transformer
- Multi-task learning
- Too many PEFT techniques
- Multiple datasets before the first experiment works

First get this pipeline working:

```text
DNABERT-2
    ↓
Full FT / LoRA / QLoRA / Adapter
    ↓
Promoter / TFBS prediction
    ↓
SHAP
    ↓
Important DNA region
    ↓
Motif analysis
    ↓
Text explanation
```

That is already a complete and defensible project.

---

# 32. Final One-Line Description

> **An explainable DNABERT-2 based framework for promoter and transcription factor binding site prediction, comparing full fine-tuning with LoRA, QLoRA and Adapter-based fine-tuning while using SHAP/LIME and biological motif analysis to provide interpretable predictions.**

# 33. Recommended Repository Name

```text
dna-xai
```

Short, memorable, and broad enough that the repository does not become misleading if the project later expands from promoter/TFBS prediction to other regulatory DNA tasks.