# DNA-XAI: Explainable DNA Sequence Classification using DNABERT-2

This repository implements a modular, reproducible framework for **promoter region** and **transcription factor binding site (TFBS) prediction** using the DNABERT-2 genomic foundation model. 

It compares full fine-tuning (our baseline) with three Parameter-Efficient Fine-Tuning (PEFT) methods: **LoRA**, **QLoRA (4-bit)**, and **custom Bottleneck Adapters**. Predictive performance is explained at base-level resolution using **SHAP** and **LIME** local attribution models grounded in **JASPAR position weight matrices (PWMs)**.

---

## 1. Problem Statement
Regulatory elements such as promoters and TFBS control gene expression. Accurately predicting these sites from primary sequence is critical for functional genomics. However, deep learning models often act as black boxes, predicting regulatory activity without revealing the sequence motifs driving their predictions. This project addresses the dual challenge of **parameter efficiency** in fine-tuning large models and **grounded biological interpretability**.

## 2. Project Architecture
The system processes DNA sequences through the following pipeline:

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
                     │ DNABERT-2 Model  │
                     └────────┬─────────┘
                              │
               ┌──────────────┼──────────────┐
               │              │              │
               ▼              ▼              ▼
           Full FT          LoRA           QLoRA
               │              │              │
               │          Adapters           │
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
                     │   SHAP & LIME    │
                     │  Attributions    │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │   Attribution    │
                     │   Overlap IoU    │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Important DNA   │
                     │     Regions      │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Motif Analysis  │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ Biological Text  │
                     │   Explanation    │
                     └──────────────────┘
```

---

## 3. Dataset
We evaluate our methods on two primary datasets:
1. **Core Promoter Dataset** (`prom_core_all`): Consists of DNA sequences of length 70 labeled as promoters (1) or non-promoters (0).
2. **Transcription Factor Binding Site Dataset** (`tf`): Comprises 5 subfolders (0 through 4) containing sequence files of length 101 labeled as binding sites (1) or non-binding sites (0).

During preparation, files are merged and globally deduplicated to prevent data leakage before being split deterministically (80% Train, 10% Validation, 10% Test).

---

## 4. Installation
Ensure Python 3.10+ and a CUDA-capable GPU environment are available.

Clone the repository and install the dependencies:
```bash
git clone https://github.com/your-username/dna-xai.git
cd dna-xai
pip install -r requirements.txt
```

---

## 5. Usage & Teammate Integration Guide

This repository is designed for **collaborative development**. To distribute training runs across different team members:

### Step 1: Preprocess and Split Data (To be run by ONE teammate)
This step cleans raw sequences, deduplicates them globally to prevent data leakage, and saves deterministic splits:
```bash
python scripts/prepare_data.py
```
This saves pre-split CSVs in `data/processed/promoter/` and `data/processed/tf/`. Push these or copy them to your teammates' systems.

### Step 2: Distribute Training Across Teammates
Each teammate clones the repo, loads the preprocessed data, and executes their assigned method:

#### Teammate A: Full Fine-Tuning (Baseline)
Updates all model weights.
```bash
# Promoter Task
python scripts/train.py --method full --task promoter

# TFBS Task (e.g., Subfolder 0)
python scripts/train.py --method full --task tf --tf_subdir 0
```

#### Teammate B: LoRA Fine-Tuning
Trains query and value attention projection matrices.
```bash
# Promoter Task
python scripts/train.py --method lora --task promoter

# TFBS Task (e.g., Subfolder 0)
python scripts/train.py --method lora --task tf --tf_subdir 0
```

#### Teammate C: QLoRA (4-bit Quantized) Fine-Tuning
Trains LoRA adapters on top of double-quantized base model weights (Requires CUDA).
```bash
# Promoter Task
python scripts/train.py --method qlora --task promoter

# TFBS Task (e.g., Subfolder 0)
python scripts/train.py --method qlora --task tf --tf_subdir 0
```

#### Teammate D: Bottleneck Adapters
Trains custom-inserted bottleneck adapter layers while the base model is frozen.
```bash
# Promoter Task
python scripts/train.py --method adapters --task promoter

# TFBS Task (e.g., Subfolder 0)
python scripts/train.py --method adapters --task tf --tf_subdir 0
```

*Note: Saved checkpoints are stored in `experiments/` and training metrics are written to `experiments/{method}/`.*

### Step 3: Run Evaluation (Each Teammate Runs on Their Checkpoint)
After training, evaluate the checkpoint on the test set to generate metrics JSONs and curve charts:
```bash
# Example evaluation command for LoRA promoter model
python scripts/evaluate.py --checkpoint experiments/promoter_lora.pt --task promoter

# Example evaluation command for QLoRA TFBS model
python scripts/evaluate.py --checkpoint experiments/tf_0_qlora.pt --task tf --tf_subdir 0
```
This writes metrics to `results/metrics/` and creates charts in `results/plots/`.

### Step 4: Share Metrics and Compile Comparison Report
To aggregate all teammates' achievements:
1. Collect the JSON files generated by each teammate from:
   - `results/metrics/*.json`
   - `experiments/{method}/*_efficiency_stats.json`
2. Push all JSON files to the main branch or merge them into a single workspace.
3. Run the aggregation CLI to compile the final study matrices:
   ```bash
   # Compile comparison tables for Promoter prediction
   python scripts/run_all_experiments.py --task promoter --aggregate_only

   # Compile comparison tables for TFBS prediction (Subfolder 0)
   python scripts/run_all_experiments.py --task tf --tf_subdir 0 --aggregate_only
   ```
   This generates comparison reports (e.g., `results/tables/comparison_promoter.md`).

### Step 5: Verify the Codebase (Run Unit Tests)
Before pushing to production, verify all features work by running pytest:
```bash
pytest tests/
```


---

## 6. Interpretability (SHAP & LIME)
To explain a prediction, we run sequence perturbations (masking positions with 'N' to simulate base mutations), measure prediction changes, compute nucleotide-level SHAP/LIME scores, scan for JASPAR motifs, and generate a natural-language report:

```bash
# Explain sample index 0 from the test split
python scripts/explain.py --checkpoint experiments/promoter_lora.pt --task promoter --sample_idx 0

# Explain a custom sequence string
python scripts/explain.py --checkpoint experiments/promoter_lora.pt --task promoter --sequence "GCTAGCTCATCTTGCGGCTGGGCGGGGCCCAGGACTGCTGCTGCTGACCGCCTTGATAGGCTACACCGTG"
```

### Example Report Output:
```text
==================================================
DNA REGULATORY ELEMENT PREDICTION REPORT
==================================================
Prediction: Promoter
Confidence: 96.4%
SHAP/LIME Consistency Overlap (IoU): 0.75
--------------------------------------------------
IMPORTANT SEQUENCE REGIONS (SHAP)
--------------------------------------------------
Region 1: positions 27–35
  Sequence:     TATAAAAG
  SHAP Attribution: 0.3421 (positive (promotes prediction))
--------------------------------------------------
BIOLOGICAL EVIDENCE (MOTIF MATCHING)
--------------------------------------------------
Matched Motif: TATA-box
  Sequence Region: positions 27–34 ('TATAAAA')
  Matching Score:  0.92 (Threshold: 0.75)
  Database Source: JASPAR core
--------------------------------------------------
FINAL INTERPRETATION
--------------------------------------------------
The sequence is predicted to be 'Promoter' with high confidence. The model's decision heavily relied on the region containing positions 27–34. This region aligns with a known 'TATA-box' motif from the database. This biological evidence is consistent with the model's prediction.
==================================================
```

---

## 7. Repository Structure
```text
dna-xai/
│
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── config.yaml               # Training & model configuration
├── .gitignore                # Git exclusions
│
├── data/
│   └── processed/            # Deduplicated train/val/test splits (gitignored)
│
├── src/                      # Source package
│   ├── data/                 # Preprocessing, dataset loaders, splitting
│   ├── models/               # DNABERT-2 wrappers, classification head, PEFT modules
│   ├── training/             # PyTorch trainer loop, evaluators, efficiency trackers
│   ├── explainability/       # Base-level SHAP, LIME, report generators
│   ├── motifs/               # Position Weight Matrix scanning
│   └── utils/                # Metrics, logging configs, random seeds
│
├── scripts/                  # CLI execution scripts
│   ├── prepare_data.py       # Cleans and splits raw data
│   ├── train.py              # Main training interface
│   ├── evaluate.py           # Evaluates test splits
│   ├── explain.py            # Generates attributions and motif scans
│   └── run_all_experiments.py# Sequential runner & aggregator
│
├── results/                  # Results assets (gitignored)
│   ├── metrics/              # Metric JSON files
│   ├── plots/                # ROC/PR curves & confusion matrices
│   ├── explanations/         # Base attributions & text reports
│   └── tables/               # Aggregated markdown study tables
│
└── tests/                    # Unit testing suite
```

---

## 8. References
*   **DNABERT-2 Paper**: Zhou, J., et al. (2023). *DNABERT-2: Efficient and Effective Foundation Model for Multi-Species Genomes*. [arXiv:2306.15006](https://arxiv.org/html/2306.15006v2).
*   **DNABERT-2 Repository**: [MAGICS-LAB/DNABERT_2](https://github.com/MAGICS-LAB/DNABERT_2).
*   **PEFT**: Hugging Face Parameter-Efficient Fine-Tuning library.
*   **JASPAR Database**: Portales-Casamar, E., et al. (2010). *JASPAR 2010: the open-access database of transcription factor binding profiles*.

---

## 9. Limitations
*   **Local Linear Approximation**: SHAP and LIME provide local feature attributions around individual sequences but do not capture complex non-linear nucleotide interactions (epistasis).
*   **Resolution**: Quantization in QLoRA may slightly reduce the absolute precision of attribution values relative to FP32 full fine-tuning.
*   **Motif Completeness**: Our motif scanner is restricted to core consensus transcription factor matches; more extensive biological annotations require integration with high-throughput toolkits like MEME Suite.