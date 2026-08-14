# Agent Instructions — dna-xai

## 1. Project Objective

Build the complete `dna-xai` project for **explainable promoter and transcription factor binding site prediction using DNABERT-2**.

The system must:

1. Load and preprocess a DNA sequence dataset.
2. Fine-tune DNABERT-2 for the selected classification task.
3. Establish a full fine-tuning baseline.
4. Implement three parameter-efficient fine-tuning approaches:
   - LoRA
   - QLoRA
   - Adapters
5. Evaluate all approaches using identical data splits and evaluation settings.
6. Implement explainability using SHAP as the primary method.
7. Optionally implement LIME as a secondary explanation method.
8. Identify important DNA regions/k-mers/tokens contributing to predictions.
9. Connect important regions with known biological motifs where possible.
10. Generate concise, human-readable biological explanations.
11. Produce reproducible metrics, plots, tables, checkpoints, and experiment logs.
12. Keep the implementation modular so that promoter prediction and TFBS prediction can be treated as separate tasks.

The project is a **course/research project**, not a production application. Prioritize correctness, reproducibility, interpretability, and clean experimentation over unnecessary architectural complexity.

---

# 2. Core Architecture

Implement the following pipeline:

```text
DNA Sequence
     │
     ▼
Data Preprocessing
     │
     ▼
DNABERT-2 Tokenizer
     │
     ▼
DNABERT-2 Encoder
     │
     ├───────────────────────────────┐
     │                               │
     ▼                               ▼
Full Fine-Tuning              PEFT Fine-Tuning
                                    │
                         ┌──────────┼──────────┐
                         ▼          ▼          ▼
                       LoRA       QLoRA     Adapters
                         │          │          │
                         └──────────┼──────────┘
                                    │
                                    ▼
                             Classification Head
                                    │
                                    ▼
                          Prediction + Probability
                                    │
                                    ▼
                            SHAP Explainability
                                    │
                                    ▼
                           Important DNA Regions
                                    │
                                    ▼
                             Motif Identification
                                    │
                                    ▼
                         Biological Text Explanation
```

Do NOT introduce a second generative transformer unless explicitly requested later.

The primary interpretability mechanism should be **model attribution + biological motif grounding**, rather than an LLM hallucinating an explanation.

---

# 3. Important Design Principle

The final explanation must be **grounded in actual model evidence**.

Never generate an explanation such as:

> "The model predicted promoter because TATA-box is important"

unless the pipeline actually found evidence for a TATA-like motif or the attribution supports that region.

The explanation pipeline should follow:

```text
Model prediction
      ↓
Attribution
      ↓
Important region
      ↓
Sequence extracted from that region
      ↓
Motif analysis
      ↓
Evidence
      ↓
Explanation
```

If no known motif is found, explicitly state that.

Example:

```text
Prediction: Promoter
Confidence: 0.94

Important region:
positions 24–35

Motif evidence:
No known motif identified

Explanation:
The model strongly relied on positions 24–35.
However, no matching known regulatory motif was
identified in this region. Therefore, the attribution
provides model-level evidence but not a confirmed
biological motif interpretation.
```

Do not fabricate biological interpretations.

---

# 4. Repository

Repository name:

```text
dna-xai
```

Expected structure:

```text
dna-xai/
│
├── README.md
├── plan.md
├── instruction.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
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
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   ├── dataset.py
│   │   └── splitting.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dnabert.py
│   │   ├── classifier.py
│   │   └── peft_models.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── experiment.py
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py
│   │   ├── lime_explainer.py
│   │   └── explanation.py
│   │
│   ├── motifs/
│   │   ├── __init__.py
│   │   └── motif_analysis.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       ├── logging.py
│       └── seed.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   └── run_all_experiments.py
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
│   ├── explanations/
│   └── tables/
│
└── tests/
    ├── test_data.py
    ├── test_model.py
    ├── test_metrics.py
    └── test_explainability.py
```

Do not create unnecessary files.

---

# 5. Technology Stack

Use Python.

Preferred libraries:

```text
Python
PyTorch
Hugging Face Transformers
PEFT
bitsandbytes
datasets
scikit-learn
SHAP
LIME
pandas
numpy
matplotlib
seaborn
```

For motif analysis, use an appropriate biological sequence/motif library where useful, such as:

```text
Biopython
```

Use JASPAR or another established motif database if the selected dataset/task supports it.

Do not download external datasets automatically without documenting the source.

---

# 6. DNABERT-2

Use the official DNABERT-2 implementation/model configuration wherever possible.

The base paper:

```text
https://arxiv.org/html/2306.15006v2
```

Official repository:

```text
https://github.com/MAGICS-LAB/DNABERT_2
```

Do not replace DNABERT-2 with ordinary BERT, BERT-base, or a generic language model.

Before implementing the model wrapper, inspect the current DNABERT-2 API/model architecture and adapt the code to the actual installed Transformers/model interface.

Do not assume model class names without checking the implementation.

---

# 7. Dataset Strategy

The agent must first inspect the available dataset before implementing the final training pipeline.

Determine:

- Dataset source
- Sequence length
- Number of samples
- Label format
- Positive/negative distribution
- Duplicate sequences
- Invalid nucleotides
- Whether promoter and TFBS labels are available
- Whether sequences overlap
- Whether train/test splits already exist

Do not blindly assume that one dataset supports both promoter and TFBS prediction.

If only promoter data is available, implement promoter classification first and structure the code so TFBS can be added later.

If only TFBS data is available, do the same for TFBS.

---

# 8. Data Preprocessing

Implement a deterministic preprocessing pipeline.

Minimum steps:

```text
Raw sequence
    ↓
Convert to uppercase
    ↓
Validate DNA characters
    ↓
Handle N / ambiguous bases
    ↓
Remove duplicates where appropriate
    ↓
Normalize labels
    ↓
Train/validation/test split
    ↓
DNABERT-2 tokenization
```

Supported DNA characters should be explicitly documented.

Do not silently discard large portions of the dataset.

Log:

```text
Initial samples
Removed samples
Final samples
Positive samples
Negative samples
Class ratio
Sequence length statistics
```

---

# 9. Data Leakage Prevention

This is extremely important.

DNA datasets may contain highly similar or duplicated sequences.

Before splitting:

- identify exact duplicates;
- investigate highly similar sequences if feasible;
- ensure identical sequences do not appear across train and test.

Never report inflated performance caused by data leakage.

Use a fixed random seed.

Store the split indices/files so every experiment uses exactly the same split.

---

# 10. Classification Setup

Start with binary classification.

Expected output:

```text
0 = negative
1 = positive
```

Use a classification head on top of the DNABERT-2 representation.

Architecture can initially be:

```text
DNABERT-2
    ↓
Sequence representation
    ↓
Dropout
    ↓
Linear
    ↓
Activation
    ↓
Classification
```

Do not add unnecessary layers unless experiments justify them.

---

# 11. Baseline Experiment

First implement:

```text
DNABERT-2
+
Full Fine-Tuning
+
Classification Head
```

Record:

- accuracy
- precision
- recall
- F1
- AUROC
- confusion matrix
- training time
- inference time
- trainable parameter count
- GPU memory

Do not implement LoRA/QLoRA/Adapters before the baseline works.

---

# 12. LoRA

Implement LoRA using the Hugging Face PEFT library.

Identify suitable target modules in DNABERT-2 after inspecting the actual architecture.

Do not blindly use target module names from an unrelated BERT implementation.

Configurable parameters should include:

```text
r
alpha
dropout
target_modules
```

Log all LoRA configuration.

---

# 13. QLoRA

Implement QLoRA using:

```text
4-bit quantization
+
LoRA
```

Use an appropriate bitsandbytes configuration.

The code must gracefully detect whether:

- CUDA is available;
- 4-bit quantization is supported;
- bitsandbytes is installed.

If QLoRA cannot run on the current machine, provide a clear error message rather than silently falling back to full precision.

Record:

```text
Quantization configuration
Compute dtype
Storage dtype
Trainable parameters
GPU memory
Training time
Performance
```

---

# 14. Adapters

Implement an adapter-based PEFT approach.

Use a maintained adapter implementation compatible with the DNABERT-2/Transformers architecture.

If a dependency introduces compatibility problems, isolate the adapter implementation behind:

```text
src/models/peft_models.py
```

Do not contaminate the rest of the training pipeline with adapter-specific logic.

---

# 15. Fair Experimental Comparison

All four experiments must use:

```text
Same dataset
Same train/validation/test split
Same preprocessing
Same sequence length
Same evaluation metrics
Same early stopping strategy
Same random seed policy
```

Where possible, use comparable training budgets.

Do not tune one model heavily and leave another at defaults.

Create a central configuration file.

Example:

```yaml
seed: 42

model:
  name: DNABERT-2
  max_length: 512

training:
  batch_size: 8
  learning_rate: 2e-5
  epochs: 5
  weight_decay: 0.01

peft:
  lora:
    r: 8
    alpha: 16
    dropout: 0.1

  qlora:
    bits: 4

  adapters:
    bottleneck: 64
```

Values are examples only. Tune them based on the dataset and hardware.

---

# 16. Evaluation

Implement a reusable evaluation module.

Required metrics:

```text
Accuracy
Precision
Recall
F1
AUROC
```

Also generate:

```text
Confusion Matrix
ROC Curve
Precision-Recall Curve
```

For imbalanced datasets, explicitly discuss why accuracy may be misleading.

Prefer macro F1 or class-wise metrics where appropriate.

---

# 17. Efficiency Evaluation

For every model, record:

```text
Total parameters
Trainable parameters
Percentage trainable
Peak GPU memory
Training duration
Average inference latency
```

Create a comparison table.

Example:

```text
Method | F1 | AUROC | Trainable % | GPU MB | Training Time
```

This comparison is one of the main contributions of the project.

---

# 18. SHAP Explainability

SHAP should be the **primary explainability method**.

The implementation must be compatible with the actual DNABERT-2 input pipeline.

Do not simply apply tabular SHAP to an arbitrary embedding vector and claim that individual nucleotides are being explained.

The explanation must map back to meaningful DNA input units.

Possible explanation units:

```text
nucleotide
k-mer
DNABERT token
sequence region
```

Prefer the representation that gives the clearest biological interpretation.

The final output should identify:

```text
Region
Sequence
Attribution
Contribution direction
```

Example:

```text
Region: 24–31
Sequence: TATAAA
Contribution: +0.31
```

---

# 19. LIME

LIME is optional but recommended as a secondary validation method.

Use perturbations of meaningful sequence units.

Possible units:

```text
k-mers
sequence windows
DNA tokens
```

Do not treat arbitrary embedding dimensions as biological features.

The goal is to answer:

> Does LIME identify approximately the same important sequence regions as SHAP?

---

# 20. Explanation Consistency

For selected examples, calculate the overlap between SHAP and LIME important regions.

Possible metric:

```text
Region overlap / Intersection-over-Union
```

For example:

```text
SHAP:
20–35

LIME:
23–34

Overlap:
High
```

This gives a quantitative explanation-consistency measure.

---

# 21. Motif Analysis

After obtaining important regions:

```text
Attribution
    ↓
Top region
    ↓
Extract DNA sequence
    ↓
Motif matching
```

Use an established motif source.

Potential sources:

```text
JASPAR
HOCOMOCO
```

Do not invent motif names.

If no significant motif is detected:

```text
No known motif detected.
```

This is an acceptable result.

---

# 22. Biological Explanation Generator

Implement a deterministic explanation generator.

Input:

```text
prediction
confidence
important_regions
attribution_scores
motif_results
```

Output:

```text
Prediction:
Promoter

Confidence:
94.2%

Important region:
Positions 31–38

Sequence:
TATAAA

Evidence:
High positive SHAP contribution

Biological evidence:
TATA-like motif detected

Explanation:
The model strongly relied on positions 31–38.
This region contains a TATA-like motif, providing
biological evidence consistent with promoter activity.
```

Use templates rather than an LLM initially.

The explanation generator must never invent:

- motifs
- transcription factors
- biological functions
- positions
- confidence values

Everything in the explanation must originate from computed evidence.

---

# 23. Explanation for Negative Cases

Explainability must work for both positive and negative predictions.

Example:

```text
Prediction:
Non-Promoter

Confidence:
91%

Important regions:
...

Interpretation:
The model assigned negative contribution to the
identified regions and did not find strong evidence
for known promoter-associated motifs.
```

Do not claim that absence of a motif proves that a sequence is not a promoter.

Use cautious language.

---

# 24. Correct vs Incorrect Predictions

Run explainability on four categories:

```text
True Positive
True Negative
False Positive
False Negative
```

This is important for the report.

Especially analyze false positives and false negatives.

Ask:

- Which regions influenced the error?
- Was there a known motif?
- Did SHAP/LIME focus on irrelevant regions?
- Does the model confuse similar regulatory patterns?

This will provide a much stronger analysis than only explaining correct predictions.

---

# 25. Visualizations

Create useful plots.

Required:

1. Training/validation loss
2. Training/validation F1
3. Confusion matrices
4. ROC curves
5. Precision-recall curves
6. Fine-tuning comparison
7. Trainable parameter comparison
8. GPU memory comparison
9. SHAP sequence attribution visualization
10. SHAP/LIME region comparison

Avoid creating dozens of meaningless plots.

---

# 26. Reproducibility

Set seeds for:

```text
Python
NumPy
PyTorch
Transformers
Dataset splitting
```

Save:

```text
configuration
random seed
model name
dataset version
training arguments
PEFT configuration
environment information
```

Each experiment should be reproducible from a command.

Example:

```bash
python scripts/train.py --method full
python scripts/train.py --method lora
python scripts/train.py --method qlora
python scripts/train.py --method adapters
```

---

# 27. Experiment Runner

Implement:

```bash
python scripts/run_all_experiments.py
```

It should sequentially execute:

```text
1. Full fine-tuning
2. LoRA
3. QLoRA
4. Adapters
5. Evaluation
6. Results aggregation
```

Do not automatically run expensive experiments without a command-line confirmation/configuration.

Support individual experiments as well.

---

# 28. Results Storage

Never overwrite results.

Use:

```text
results/
    metrics/
    plots/
    explanations/
```

Example:

```text
results/metrics/full.json
results/metrics/lora.json
results/metrics/qlora.json
results/metrics/adapters.json
```

Explanations:

```text
results/explanations/sample_001.json
```

Plots:

```text
results/plots/model_comparison.png
results/plots/shap_sample_001.png
```

---

# 29. Testing

Create lightweight tests for:

### Data

- invalid DNA handling
- label conversion
- deterministic splitting

### Model

- forward pass
- output shape
- prediction range

### Metrics

- metric calculation
- binary labels

### Explainability

- explanation output structure
- attribution dimensions
- region extraction
- no fabricated motif output

Tests should run without requiring the full dataset/GPU whenever possible.

---

# 30. Command-Line Interface

Provide clear commands.

Examples:

```bash
# Prepare data
python scripts/prepare_data.py

# Train baseline
python scripts/train.py --method full

# Train LoRA
python scripts/train.py --method lora

# Train QLoRA
python scripts/train.py --method qlora

# Train Adapters
python scripts/train.py --method adapters

# Evaluate
python scripts/evaluate.py --checkpoint <path>

# Explain one sequence
python scripts/explain.py --checkpoint <path> --sequence "<DNA_SEQUENCE>"

# Run complete experiment
python scripts/run_all_experiments.py
```

If argument names differ, document the actual interface in README.md.

---

# 31. README Requirements

README.md must contain:

1. Project title
2. Problem statement
3. Motivation
4. Architecture
5. Dataset
6. Installation
7. Usage
8. Training commands
9. Evaluation
10. Explainability
11. Example prediction
12. Results table
13. Repository structure
14. References
15. Limitations

Include the DNABERT-2 paper and official repository as references.

---

# 32. Coding Standards

Follow these rules:

- Python type hints where practical.
- Small modular functions.
- Avoid giant scripts.
- Avoid duplicated training logic.
- Use configuration rather than hard-coded hyperparameters.
- Use meaningful variable names.
- Add docstrings to public functions/classes.
- Log important training information.
- Fail loudly on invalid configurations.
- Never silently swallow exceptions.
- Avoid unnecessary abstractions.

Do not optimize prematurely.

First make the simplest correct implementation work.

---

# 33. Hardware Awareness

The project may be executed on a limited GPU.

Therefore:

- support configurable batch size;
- support gradient accumulation;
- support mixed precision;
- support gradient checkpointing where useful;
- support QLoRA for memory reduction;
- avoid loading multiple large models simultaneously;
- release GPU memory between experiments.

Do not assume a specific GPU.

Detect available hardware automatically.

---

# 34. Development Order

Follow this exact order.

## Step 1

Inspect the repository and available dataset.

## Step 2

Set up dependencies and configuration.

## Step 3

Implement dataset preprocessing.

## Step 4

Implement DNABERT-2 loading and tokenization.

## Step 5

Implement classification head.

## Step 6

Run a tiny forward-pass test.

## Step 7

Implement full fine-tuning.

## Step 8

Run a small training experiment.

## Step 9

Implement evaluation.

## Step 10

Implement LoRA.

## Step 11

Implement QLoRA.

## Step 12

Implement Adapters.

## Step 13

Run all PEFT experiments.

## Step 14

Aggregate results.

## Step 15

Implement SHAP.

## Step 16

Verify SHAP attribution maps back to meaningful DNA regions.

## Step 17

Implement LIME if computationally practical.

## Step 18

Implement motif analysis.

## Step 19

Implement biological explanation generation.

## Step 20

Analyze TP/TN/FP/FN examples.

## Step 21

Generate final plots/tables.

## Step 22

Complete README and documentation.

## Step 23

Run tests.

## Step 24

Run a final end-to-end smoke test.

---

# 35. Minimum Viable Version

If time or GPU resources are limited, prioritize:

```text
DNABERT-2
   ↓
Full Fine-Tuning
   ↓
LoRA
   ↓
QLoRA
   ↓
Adapters
   ↓
Evaluation
   ↓
SHAP
   ↓
Motif analysis
   ↓
Text explanation
```

LIME is secondary.

Do not sacrifice the core pipeline to implement unnecessary features.

---

# 36. Definition of Done

The project is complete only when:

- [ ] Dataset preprocessing works.
- [ ] Dataset statistics are documented.
- [ ] Train/validation/test split is reproducible.
- [ ] DNABERT-2 loads correctly.
- [ ] Baseline full fine-tuning works.
- [ ] LoRA works.
- [ ] QLoRA works.
- [ ] Adapters work.
- [ ] All methods use the same evaluation split.
- [ ] Accuracy, precision, recall, F1 and AUROC are reported.
- [ ] Trainable parameter counts are reported.
- [ ] GPU memory/training time are measured where available.
- [ ] SHAP explanation works.
- [ ] Important DNA regions can be mapped back to the input sequence.
- [ ] LIME is implemented or explicitly documented as optional/not feasible.
- [ ] Motif analysis works.
- [ ] Biological explanations are grounded in actual evidence.
- [ ] False positives and false negatives are analyzed.
- [ ] Results are saved reproducibly.
- [ ] README is complete.
- [ ] Tests pass.
- [ ] End-to-end inference works.

---

# 37. Important Restrictions

Do NOT:

- fabricate experimental results;
- fabricate biological motifs;
- fabricate dataset statistics;
- claim SHAP explains biological causality;
- claim attention automatically equals explanation;
- compare models using different test sets;
- use accuracy alone on an imbalanced dataset;
- silently replace DNABERT-2 with another model;
- silently fall back from QLoRA to full precision;
- hard-code machine-specific paths;
- commit datasets or model checkpoints unnecessarily;
- introduce an LLM-generated explanation without grounding it in model evidence.

---

# 38. Interpretation Philosophy

The project should distinguish clearly between:

### Model evidence

```text
SHAP contribution
LIME importance
Prediction probability
```

and:

### Biological evidence

```text
Known DNA motif
Motif database match
Known regulatory annotation
```

and:

### Interpretation

```text
Human-readable explanation
```

These are not interchangeable.

The final system should communicate uncertainty.

For example:

> "The model strongly relied on this region, and the region contains a motif associated with promoter activity."

is valid.

Avoid:

> "This motif caused the sequence to be a promoter."

unless there is experimental causal evidence.

---

# 39. Final Expected Pipeline

The final implementation should achieve:

```text
                    ┌──────────────────┐
                    │   DNA Sequence   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Preprocessing   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    DNABERT-2     │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
          Full FT          LoRA           QLoRA
                             │
                             │
                         Adapters
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Classification   │
                    │      Head        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Prediction +     │
                    │ Confidence       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │      SHAP        │
                    │   Explainability │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Important DNA    │
                    │     Regions      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Motif Analysis   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Biological Text  │
                    │   Explanation    │
                    └──────────────────┘
```

---

# 40. Final Goal

The finished `dna-xai` repository should demonstrate that:

> **DNABERT-2 can be adapted for regulatory DNA sequence classification using multiple parameter-efficient fine-tuning strategies, while SHAP/LIME-based attribution and motif analysis can provide grounded, interpretable evidence for individual predictions.**

The primary success criterion is **not simply achieving the highest accuracy**.

The project should demonstrate a meaningful trade-off between:

```text
Predictive Performance
        +
Parameter Efficiency
        +
Computational Efficiency
        +
Interpretability
        +
Biological Plausibility
```

That combination is the core of `dna-xai`.