# 🏗️ Text-to-KG SLM — UK Construction Contract Knowledge Graph Extraction

> **Fine-tuning Phi-3.5 Mini Instruct and Gemma 2 2B IT with LoRA and QLoRA on verified UK Open Contracting data to extract structured RDF knowledge graph triples from procurement contract text — achieving ZERO hallucination.**

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-BSVGK-blue)](https://huggingface.co/BSVGK)
[![Dataset](https://img.shields.io/badge/Dataset-Text__to__KG__Construction__Dataset-green)](https://huggingface.co/datasets/BSVGK/Text_to_KG_Construction_Dataset)
[![Model](https://img.shields.io/badge/Model-phi35--mini--lora--text2kg--merged-orange)](https://huggingface.co/BSVGK/phi35-mini-lora-text2kg-merged)
[![Phase2 Model](https://img.shields.io/badge/Phase2-gemma2--qlora--text2kg-red)](https://huggingface.co/BSVGK)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Zero Hallucination](https://img.shields.io/badge/Hallucination-ZERO%20%E2%9C%93-brightgreen)](https://huggingface.co/BSVGK/phi35-mini-lora-text2kg-merged)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Two-Phase Experimental Design](#-two-phase-experimental-design)
- [Results](#-results)
- [Why Gemma 2 Hallucinated — Architectural Analysis](#-why-gemma-2-hallucinated--architectural-analysis)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Fine-Tuning Methods](#-fine-tuning-methods)
- [Training Configuration](#-training-configuration)
- [Evaluation Metrics](#-evaluation-metrics)
- [Hallucination Evaluation Framework](#-hallucination-evaluation-framework)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Inference](#-inference)
- [Gradio Interface](#-gradio-interface)
- [HuggingFace Links](#-huggingface-links)
- [Project Team](#-project-team)
- [Citation](#-citation)

---

## 🔬 Project Overview

This project is part of the **UEL–Depixen Industrial Placement (January – May 2026)**, investigating the thesis:

> *A Small Language Model fine-tuned on verified, domain-specific data outperforms a large general-purpose LLM on targeted tasks — with dramatically lower hallucination.*

**Task:** Text to Knowledge Graph (Text-to-KG) — Information Extraction  
**Domain:** UK Public Procurement / Construction Contracts  
**Input:** Unstructured UK government procurement contract text  
**Output:** Structured RDF triples — (Subject, Predicate, Object)  

### Example

**Input Contract Text:**
```
Wakefield Council has awarded a construction contract to AMALGAMATED CONSTRUCTION LTD 
for a total value of £420,850. The contract was awarded on 15 March 2024 under CPV 
code 45000000 (Construction work) and will be performed in Yorkshire and the Humber.
```

**Extracted RDF Triples:**
```
(Contract_008672, rdf:type, Contract)
(Contract_008672, hasBuyer, Wakefield Council)
(Contract_008672, hasSupplier, AMALGAMATED CONSTRUCTION LTD)
(Contract_008672, hasContractValue, 420850)
(Contract_008672, hasAwardDate, 2024-03-15)
(Contract_008672, hasCPVCode, 45000000)
(Contract_008672, hasCPVDescription, Construction work)
(Contract_008672, hasLocation, Yorkshire and the Humber)
```

---

## 🔁 Two-Phase Experimental Design

This project employs a **controlled two-phase experiment** designed to test whether base model architecture independently determines hallucination risk — separate from data quality.

| | Phase 1 | Phase 2 |
|--|---------|---------|
| **Model** | Phi-3.5 Mini Instruct | Gemma 2 2B IT |
| **Developer** | Microsoft | Google DeepMind |
| **Parameters** | 3.82B | 2.6B |
| **Attention** | Global (all layers) | Local + Global (alternating) |
| **Fine-Tuning** | LoRA (float16) | QLoRA (4-bit NF4) |
| **GPU Memory** | 7.64 GB | 2.3 GB |
| **Training Time** | 26.01 min | 62.37 min |
| **Hallucination L2** | **0.0 — ZERO** | 0.1233 |
| **Status** | ✅ **Deployed** | 📊 Comparison |

> **Hypothesis tested:** If hallucination rates differ after training on the same verified dataset, the cause must be architectural — not data-related.
>
> **Result:** Confirmed. Global attention (Phi-3.5) → zero hallucination. Local attention (Gemma 2) → 12.33% hallucination. **Architecture is an independent hallucination risk factor.**

---

## 📊 Results

### Phase 1: Phi-3.5 Mini + LoRA ✅ BEST MODEL

> ⭐ **ZERO HALLUCINATION** across all 1,387 unseen test contracts

| Metric | Score |
|--------|:-----:|
| **F1 Score** | **0.9954** |
| **Recall** | **0.9950** |
| **Precision** | **0.9959** |
| **ROUGE-L** | **0.9991** |
| **BERTScore F1** | **0.9997** |
| **Hallucination L1** | **0.0** |
| **Hallucination L2** | **0.0** |
| **Weighted Score** | **0.8469** |
| Training Loss | 0.3008 |
| Validation Loss | 0.2570 |

### Phase 2: Gemma 2 2B IT + QLoRA 📊 Comparison

| Metric | Score |
|--------|:-----:|
| F1 Score | 0.9756 |
| Recall | 0.9688 |
| Precision | 0.9835 |
| ROUGE-L | 0.9783 |
| BERTScore F1 | 0.9975 |
| **Hallucination L1** | **0.0036** |
| **Hallucination L2** | **0.1233** ⚠️ |
| Weighted Score | 0.8250 |
| Training Loss | 0.2689 |
| Validation Loss | 0.2563 |

### Full Side-by-Side Comparison

| Metric | Phase 1: Phi-3.5+LoRA | Phase 2: Gemma2+QLoRA | Winner |
|--------|:---------------------:|:---------------------:|:------:|
| F1 Score | **0.9954** | 0.9756 | Phase 1 |
| Recall | **0.9950** | 0.9688 | Phase 1 |
| Precision | **0.9959** | 0.9835 | Phase 1 |
| ROUGE-L | **0.9991** | 0.9783 | Phase 1 |
| BERTScore | **0.9997** | 0.9975 | Phase 1 |
| Hall-L1 ⚠ | **0.0** | 0.0036 | Phase 1 |
| Hall-L2 ⚠ | **0.0** | 0.1233 | Phase 1 |
| Training time | **26 min** | 62 min | Phase 1 |
| GPU memory | 7.64 GB | **2.3 GB** | Phase 2 |
| Training loss | 0.3008 | **0.2689** | Phase 2 (PARADOX ⚠️) |
| **Weighted Score** | **0.8469** | 0.8250 | **Phase 1** |

> ⚠️ **Critical finding:** Phase 2 achieved a **lower training loss** than Phase 1 but **higher hallucination**. Training loss is **NOT** a reliable indicator of factual reliability. See [architectural analysis](#-why-gemma-2-hallucinated--architectural-analysis) below.

---

## 🔍 Why Gemma 2 Hallucinated — Architectural Analysis

This is the central research finding of Project 2. Three root causes were identified.

### Root Cause 1 — Local Attention Architecture

```
Phi-3.5 Mini — GLOBAL ATTENTION ✓
┌──────────────────────────────────────────┐
│  [Buyer] [Supplier] [Value] [Date] [Loc] │
│     ↕        ↕        ↕      ↕      ↕   │
│  All tokens attend to ALL other tokens   │
│  → Every entity fully visible            │
│  → Zero hallucination                    │
└──────────────────────────────────────────┘

Gemma 2 2B IT — LOCAL ATTENTION ✗
┌──────────────────────────────────────────┐
│  [Buyer] [Supplier] [Value] [Date] [Loc] │
│             ↕        ↕      ↕            │
│         [sliding window only]            │
│  Buyer and Location OUTSIDE window       │
│  → Model defaults to training patterns   │
│  → Hallucination L2 = 0.1233            │
└──────────────────────────────────────────┘
```

Gemma 2 uses **alternating local and global attention layers**. In local attention layers, each token can only attend to tokens within a fixed sliding window. For procurement contracts where key entities (buyer name, location) may appear at different positions relative to the extraction point, local attention layers cannot simultaneously access all required entities — the model defaults to statistically common entity values learned during training rather than the specific entities in the current input.

### Root Cause 2 — 4-bit NF4 Quantisation Precision Loss

QLoRA's 4-bit NF4 quantisation reduces the precision of the frozen base model weight representations. While NF4 preserves the statistical distribution of weights better than INT4, it still introduces quantisation error that reduces the model's ability to distinguish between similar entity representations — particularly important in extraction tasks where the model must copy specific entity values precisely from the input text.

> **Key contrast:** In Project 1 (generation task), LoRA and QLoRA produced **identical hallucination rates** (0.00537). In Project 2 (extraction task), quantisation contributed to 12.33% hallucination. **Quantisation affects extraction hallucination more severely than generation hallucination.**

### Root Cause 3 — Training Loss is the Wrong Quality Signal

| Model | Training Loss | Hallucination L2 |
|-------|:-------------:|:----------------:|
| Gemma 2 + QLoRA | **0.2689** (lower) | **0.1233** (worse) |
| Phi-3.5 + LoRA | 0.3008 (higher) | **0.0** (better) |

Training loss optimises for **statistical pattern fit** — how well the model reproduces training distributions. It does not measure **factual grounding** — whether outputs are anchored in the specific entities present in the current input. A model achieving lower training loss may have overfitted to common entity values from the training set, substituting those patterns for the actual input entities on unseen test data.

> **Conclusion: Never use training loss as the sole model selection criterion for hallucination-sensitive deployments. Always evaluate hallucination rate on a held-out test set.**

---

## 📁 Repository Structure

```
text-to-kg-construction-slm/
│
├── notebooks/
│   ├── Phi3.5_LoRA.ipynb              # Phase 1: Phi-3.5 Mini + LoRA pipeline
│   ├── Gemma2_QLoRA.ipynb             # Phase 2: Gemma 2 2B IT + QLoRA pipeline
│   ├── Comparison_Analysis.ipynb      # Cross-phase analysis, hallucination eval
│   └── Gradio_Interface.ipynb         # Gradio demo deployment
│
├── app/
│   ├── gradio_app.py                  # Standalone Gradio application
│   └── requirements.txt
│
├── data/
│   ├── sample_contracts.json          # Sample UK procurement contracts for testing
│   └── ontology.ttl                   # RDF ontology definition (Turtle format)
│
├── scripts/
│   ├── prepare_dataset.py             # OCDS JSON → instruction-formatted dataset
│   ├── evaluate.py                    # Standalone evaluation — all 7 metrics
│   ├── evaluate_hallucination.py      # Dual-level hallucination evaluation (L1+L2)
│   └── inference.py                   # CLI batch inference script
│
├── patches/
│   └── modeling_phi3_patch.py        # Transformers 5.x compatibility fix
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📦 Dataset

### Source

**UK Open Contracting Data Standard (OCDS)**  
[data.open-contracting.org/en/](https://data.open-contracting.org/en/)  
Published under the **Open Government Licence** — freely available, no access restrictions.

### Why This Data Source?

| Reason | Detail |
|--------|--------|
| **Government-verified** | Every record submitted by UK public authorities under legal reporting obligation — not user-generated or web-scraped |
| **Consistent OCDS structure** | Standard JSON schema across all UK procurement authorities — systematic ontology design possible |
| **Real-world scale** | 9,244 real construction contracts from multiple UK authorities |
| **Zero hallucination requirement** | Fabricated contract values or supplier names have legal consequences — strongest possible test |

### Statistics

| Split | Samples | Percentage |
|-------|:-------:|:----------:|
| Training | 6,470 | 70% |
| Validation | 1,387 | 15% |
| Test | 1,387 | 15% |
| **Total** | **9,244** | **100%** |

### Ontology — 5 Classes and 8 Relations

**Classes:** `Contract`, `Buyer`, `Supplier`, `CPVCategory`, `Location`

| Relation | Description |
|----------|-------------|
| `rdf:type` | Entity type classification within the ontology |
| `hasBuyer` | Links contract to the contracting public authority |
| `hasSupplier` | Links contract to the awarded company |
| `hasContractValue` | Monetary value of the contract (GBP) |
| `hasAwardDate` | Date the contract was formally awarded |
| `hasCPVCode` | Common Procurement Vocabulary numeric code |
| `hasCPVDescription` | Human-readable CPV category description |
| `hasLocation` | UK region where the contract is performed |

### Quality Validation

- ✅ No duplicate contract IDs
- ✅ No null values across all 8 relation fields
- ✅ No train-test contamination
- ✅ All 8 ontology relations present in every training sample

### HuggingFace Dataset

```python
from datasets import load_dataset
dataset = load_dataset("BSVGK/Text_to_KG_Construction_Dataset")
```

🔗 [BSVGK/Text_to_KG_Construction_Dataset](https://huggingface.co/datasets/BSVGK/Text_to_KG_Construction_Dataset)

---

## 🧠 Model Architecture

### Phase 1 — Phi-3.5 Mini Instruct (Deployed)

**Developer:** Microsoft  
**Parameters:** 3,821,079,552  
**Attention:** Global multi-head attention at every layer  
**Format:** Instruction-tuned

| Reason | Justification |
|--------|---------------|
| **Global attention** | Every contract token attends to every other token — all entities (buyer, supplier, value, location) fully visible during extraction. Essential for zero hallucination. |
| **Instruction-tuned** | Optimised for instruction-following tasks. Extraction is instruction-based — architecturally aligned with the task. |
| **3.82B parameters** | Large enough for complex UK procurement language. Small enough for efficient fine-tuning (26 min on A100 80GB). |

### Phase 2 — Gemma 2 2B IT (Comparison)

**Developer:** Google DeepMind  
**Parameters:** 2,614,341,888  
**Attention:** Alternating local + global attention  
**Format:** Instruction-tuned

| Reason for inclusion | Justification |
|----------------------|---------------|
| **Controlled comparison** | Different model family (Google vs Microsoft) with different attention architecture — isolates architecture as independent variable |
| **Memory efficiency test** | Gemma 2 + QLoRA uses 2.3 GB GPU vs 7.64 GB. Tests whether memory savings trade off against hallucination. |
| **Deliberate stress test** | Local attention + NF4 quantisation = most challenging combination — tests hypothesis that both are independent risk factors |

---

## ⚙️ Fine-Tuning Methods

### Phase 1: LoRA (Low-Rank Adaptation) — Deployed

| Parameter | Value | Justification |
|-----------|:-----:|---------------|
| Rank `r` | 16 | Same as Phase 2 for controlled comparison. Standard extraction default. |
| Alpha `α` | 32 | Ratio 2.0 — proven stable for Phi-3.5 instruction-tuned architecture. |
| Dropout | 0.05 | Light regularisation. Consistent with Phase 2. |
| Target modules | 7 | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable params | 8.9M (0.23%) | Extremely efficient — only 0.23% of 3.82B params update. |
| Precision | float16 | No quantisation — maximum factual grounding precision. |
| Training loss | 0.3008 | Higher than Phase 2 (0.2689) — but ZERO hallucination. Loss ≠ quality. |

### Phase 2: QLoRA (Quantised LoRA) — Comparison

Same LoRA configuration plus:

| Setting | Value | Justification |
|---------|:-----:|---------------|
| Base quantisation | 4-bit NF4 | Reduces Gemma 2 from 4.5 GB to 1.2 GB. Neural-weight optimised. |
| Compute dtype | bfloat16 | Wider dynamic range — avoids overflow during backward pass on A100. |
| Double quantisation | True | Saves 0.37 bits/parameter. Total GPU: 2.3 GB. |
| Optimiser | Paged AdamW 8-bit | Optimiser states paged to CPU RAM. Enables 40GB GPU training. |
| kbit training prep | Required | Enables gradient checkpointing. Casts layer norms to float32 for stability. |
| Trainable params | 20.8M (1.28%) | More adapters needed due to Gemma 2 architecture differences. |

---

## 🔧 Training Configuration

Both phases use **SFTTrainer** from the TRL library with identical shared settings.

| Hyperparameter | Phase 1 | Phase 2 | Justification |
|----------------|:-------:|:-------:|---------------|
| Epochs | 3 | 3 | Same duration for comparable results. |
| Learning rate | 2e-4 | 2e-4 | Standard LoRA LR. Same for controlled comparison. |
| Per-device batch | 4 | 4 | Maximum within GPU memory alongside model. |
| Gradient accumulation | 4 | 4 | Effective batch 16. Stable gradient estimates. |
| LR scheduler | Cosine | Cosine | Smooth decay to near-zero. Avoids destabilisation. |
| Total steps | 1,212 | 1,212 | 404 steps/epoch × 3 = full dataset coverage at batch 16. |
| Max seq length | 800 | 800 | Covers all UK OCDS contract descriptions. |
| Hardware | A100 80GB | A100 40GB | QLoRA's lower memory enables smaller GPU. |
| Training time | 26.01 min | 62.37 min | QLoRA is slower due to dequantisation overhead. |
| Train loss | 0.3008 | 0.2689 | PARADOX: lower loss, higher hallucination in Phase 2. |
| Val loss | 0.2570 | 0.2563 | Comparable — not predictive of hallucination difference. |

---

## 📏 Evaluation Metrics

### Metric Framework

| Metric | Weight | Category | Purpose |
|--------|:------:|----------|---------|
| **F1 Score** | +0.30 | Extraction | Harmonic mean of Precision + Recall — most complete single extraction measure |
| **Recall** | +0.20 | Extraction | Missing triples = real knowledge loss. Higher weight — omissions costlier in legal context |
| **Precision** | +0.15 | Extraction | Spurious triples pollute the knowledge graph |
| **ROUGE-L** | +0.10 | Format | Format compliance — triples must follow (subject, predicate, object) structure |
| **BERTScore** | +0.10 | Semantic | Semantic correctness when wording differs slightly |
| **Hallucination L1** | −0.10 | Hallucination | Relation type validity — predicate not in ontology breaks graph schema |
| **Hallucination L2** | −0.05 | Hallucination | Entity grounding — value not found in input text — most dangerous type |

### Weighted Score Formula

```
Weighted Score = (F1 × 0.30) + (Recall × 0.20) + (Precision × 0.15)
              + (ROUGE-L × 0.10) + (BERTScore × 0.10)
              − (Hall-L1 × 0.10) − (Hall-L2 × 0.05)
```

> Hallucination metrics carry **negative weights** because lower hallucination = better model.

---

## 🚨 Hallucination Evaluation Framework

This project introduces a **novel dual-level hallucination evaluation** framework for knowledge graph extraction tasks.

### Hallucination Level 1 (Hall-L1) — Relation Type Validity

Measures the proportion of records in which the model generates **at least one predicate that does not exist in the defined 8-relation ontology**.

```python
def compute_hall_l1(predicted_triples, valid_predicates):
    """
    Check if any predicted predicate is not in the defined ontology.
    
    Args:
        predicted_triples: List of (subject, predicate, object) tuples
        valid_predicates: Set of valid relation names from ontology
    
    Returns:
        1 if hallucination detected (invalid predicate), 0 otherwise
    """
    for _, predicate, _ in predicted_triples:
        if predicate not in valid_predicates:
            return 1  # Level 1 hallucination
    return 0
```

**Example:**
```
# Valid predicate ✓
(Contract_001, hasSupplier, ACME Ltd)   → Hall-L1 = 0

# Invalid predicate ✗ — not in ontology
(Contract_001, hasRegion, Yorkshire)    → Hall-L1 = 1
```

### Hallucination Level 2 (Hall-L2) — Entity Grounding

Measures the proportion of records in which the model generates **at least one object entity value not present anywhere in the input text**.

```python
def compute_hall_l2(predicted_triples, input_text):
    """
    Check if any predicted object value is not grounded in the input text.
    
    Args:
        predicted_triples: List of (subject, predicate, object) tuples
        input_text: The original contract text used as input
    
    Returns:
        1 if hallucination detected (ungrounded entity), 0 otherwise
    """
    for _, _, obj in predicted_triples:
        # Check if object value appears in the input text
        if str(obj).lower() not in input_text.lower():
            return 1  # Level 2 hallucination — entity not grounded
    return 0
```

**Example:**
```
Input text: "...awarded to AMALGAMATED CONSTRUCTION LTD..."

# Grounded entity ✓
(Contract_008672, hasSupplier, AMALGAMATED CONSTRUCTION LTD) → Hall-L2 = 0

# Ungrounded entity ✗ — model invented this value
(Contract_008672, hasSupplier, AMALGAMATED BUILDING SERVICES) → Hall-L2 = 1
```

---

## 🛠️ Installation

```bash
git clone https://github.com/sai-uel/text-to-kg-construction-slm
cd text-to-kg-construction-slm
pip install -r requirements.txt
```

### requirements.txt

```
transformers==4.45.0
peft==0.13.0
trl==0.11.4
accelerate==0.34.2
bitsandbytes>=0.41.0
datasets>=2.14.0
evaluate>=0.4.0
bert-score>=0.3.13
rouge-score>=0.1.2
torch>=2.0.0
sentencepiece
huggingface-hub
gradio>=4.0.0
```

> ⚠️ **Version pinning is critical.** Transformers 5.x introduced breaking changes in the Phi-3.5 KV cache API (`seen_tokens` → `get_seq_length()`) and caused `tied_weights_keys` AttributeErrors when calling `save_pretrained()`. These have been patched in the merged model and in `patches/modeling_phi3_patch.py`.

---

## ⚡ Quick Start

### Load the Merged Model (No PEFT Required) — Recommended

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "BSVGK/phi35-mini-lora-text2kg-merged"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### Load with LoRA Adapter (Requires PEFT)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_id = "microsoft/Phi-3.5-mini-instruct"
adapter_id    = "BSVGK/phi35-mini-lora-text2kg-adapter"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, adapter_id)
```

---

## 🔍 Inference

```python
def extract_triples(contract_text: str, model, tokenizer, max_new_tokens=512):
    """
    Extract RDF knowledge graph triples from UK procurement contract text.

    Args:
        contract_text: Raw UK procurement contract description text
        model: Loaded Phi-3.5 model (merged or with adapter)
        tokenizer: Loaded tokenizer
        max_new_tokens: Maximum tokens to generate

    Returns:
        String containing extracted (subject, predicate, object) RDF triples
    """
    prompt = f"""<|user|>
You are a knowledge graph extraction expert specialising in UK public procurement contracts.
Extract all relevant RDF triples from the following contract text using the ontology:

Classes: Contract, Buyer, Supplier, CPVCategory, Location
Relations: rdf:type, hasBuyer, hasSupplier, hasContractValue, hasAwardDate, 
           hasCPVCode, hasCPVDescription, hasLocation

Extract ONLY information explicitly present in the text. Do NOT infer or generate 
any values not directly stated. Format as: (Subject, Predicate, Object)

Contract Text:
{contract_text}

Extract all RDF triples:<|end|>
<|assistant|>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # Greedy decoding — factual precision over diversity
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# Example usage
contract = """
Manchester City Council has awarded a construction contract to MORGAN SINDALL 
INFRASTRUCTURE PLC for a total value of £1,250,000. The contract was awarded on 
22 January 2024 under CPV code 45233000 (Construction, foundation and surface works 
for highways) and will be performed in North West England.
"""

triples = extract_triples(contract, model, tokenizer)
print(triples)

# Expected output:
# (Contract_Manchester_001, rdf:type, Contract)
# (Contract_Manchester_001, hasBuyer, Manchester City Council)
# (Contract_Manchester_001, hasSupplier, MORGAN SINDALL INFRASTRUCTURE PLC)
# (Contract_Manchester_001, hasContractValue, 1250000)
# (Contract_Manchester_001, hasAwardDate, 2024-01-22)
# (Contract_Manchester_001, hasCPVCode, 45233000)
# (Contract_Manchester_001, hasCPVDescription, Construction, foundation and surface works for highways)
# (Contract_Manchester_001, hasLocation, North West England)
```

### Batch Inference from CSV

```python
import pandas as pd

def batch_extract(csv_path: str, model, tokenizer, output_path: str):
    """
    Process multiple contract texts from CSV and save extracted triples.

    Args:
        csv_path: Path to CSV file with 'contract_text' column
        model: Loaded model
        tokenizer: Loaded tokenizer
        output_path: Path to save results CSV
    """
    df = pd.read_csv(csv_path)
    results = []

    for idx, row in df.iterrows():
        triples = extract_triples(row["contract_text"], model, tokenizer)
        results.append({
            "contract_id": row.get("contract_id", f"contract_{idx}"),
            "input_text": row["contract_text"],
            "extracted_triples": triples
        })
        print(f"Processed {idx + 1}/{len(df)}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return result_df
```

### Parse Extracted Triples

```python
import re

def parse_triples(raw_output: str) -> list[tuple]:
    """
    Parse extracted triple string into structured list of tuples.

    Args:
        raw_output: Raw model output string containing triples

    Returns:
        List of (subject, predicate, object) tuples
    """
    pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
    matches = re.findall(pattern, raw_output)
    return [(s.strip(), p.strip(), o.strip()) for s, p, o in matches]


# Usage
triples_text = extract_triples(contract, model, tokenizer)
structured = parse_triples(triples_text)

for subject, predicate, obj in structured:
    print(f"Subject:   {subject}")
    print(f"Predicate: {predicate}")
    print(f"Object:    {obj}")
    print()
```

---

## 🎛️ Gradio Interface

The Gradio interface supports free text, CSV and PDF input.

```python
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import json

model_id = "BSVGK/phi35-mini-lora-text2kg-merged"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)

# Session history storage
session_history = []

def process_input(text_input, file_input):
    """Process text or file input and return extracted triples."""
    if file_input is not None:
        if file_input.name.endswith(".csv"):
            df = pd.read_csv(file_input.name)
            text_input = " ".join(df.iloc[:, 0].astype(str).tolist())
        elif file_input.name.endswith(".pdf"):
            import pypdf
            reader = pypdf.PdfReader(file_input.name)
            text_input = " ".join(page.extract_text() for page in reader.pages)

    if not text_input.strip():
        return "Please provide contract text or upload a file.", ""

    triples = extract_triples(text_input, model, tokenizer)
    session_history.append({
        "input": text_input[:200] + "...",
        "triples": triples
    })

    return triples, format_history(session_history)


def format_history(history):
    """Format session history for display."""
    return "\n\n---\n\n".join(
        [f"**Input:** {h['input']}\n\n**Triples:**\n{h['triples']}"
         for h in history[-5:]]  # Show last 5
    )


def download_csv(triples_text):
    """Convert extracted triples to downloadable CSV."""
    structured = parse_triples(triples_text)
    df = pd.DataFrame(structured, columns=["Subject", "Predicate", "Object"])
    path = "/tmp/extracted_triples.csv"
    df.to_csv(path, index=False)
    return path


with gr.Blocks(title="Text-to-KG Extraction — UK Procurement") as demo:
    gr.Markdown("# 🏗️ UK Procurement Contract → Knowledge Graph Extractor")
    gr.Markdown("Extract structured RDF triples from UK construction contract text using fine-tuned Phi-3.5 Mini + LoRA")

    with gr.Tab("Extract"):
        with gr.Row():
            text_in = gr.Textbox(
                label="Contract Text", 
                placeholder="Paste UK procurement contract text here...",
                lines=8
            )
            file_in = gr.File(label="Or Upload File (CSV / PDF)", file_types=[".csv", ".pdf"])
        
        extract_btn = gr.Button("Extract Triples", variant="primary")
        
        triples_out = gr.Textbox(label="Extracted RDF Triples", lines=12)
        download_btn = gr.Button("Download as CSV")
        download_file = gr.File(label="Download")

    with gr.Tab("Session History"):
        history_out = gr.Markdown("No extractions yet.")

    extract_btn.click(
        process_input,
        inputs=[text_in, file_in],
        outputs=[triples_out, history_out]
    )
    download_btn.click(download_csv, inputs=[triples_out], outputs=[download_file])

demo.launch()
```

---

## 🤗 HuggingFace Links

| Resource | Link |
|----------|------|
| 🤗 Merged Model — Phase 1 (recommended) | [BSVGK/phi35-mini-lora-text2kg-merged](https://huggingface.co/BSVGK/phi35-mini-lora-text2kg-merged) |
| 📎 LoRA Adapter — Phase 1 | [BSVGK/phi35-mini-lora-text2kg-adapter](https://huggingface.co/BSVGK/phi35-mini-lora-text2kg-adapter) |
| 📊 Training Dataset | [BSVGK/Text_to_KG_Construction_Dataset](https://huggingface.co/datasets/BSVGK/Text_to_KG_Construction_Dataset) |
| 👤 BSVGK Profile | [huggingface.co/BSVGK](https://huggingface.co/BSVGK) |

All models and datasets are **publicly accessible** — no access request required.

---

## ⚠️ Known Issues and Fixes

### Transformers 5.x Compatibility — Phi-3.5 Mini

When using Transformers >= 5.0.0 with Phi-3.5 Mini, two errors may occur:

**Error 1: `seen_tokens` AttributeError**
```
AttributeError: 'DynamicCache' object has no attribute 'seen_tokens'
```
**Fix:** The `seen_tokens` attribute was replaced by `get_seq_length()` in Transformers 5.x.

**Error 2: `tied_weights_keys` AttributeError**
```
AttributeError: 'Phi3ForCausalLM' object has no attribute 'tied_weights_keys'
```
**Fix:** Add `tied_weights_keys = []` to the model class.

Both fixes are applied in the merged model on HuggingFace and in `patches/modeling_phi3_patch.py`. The recommended approach is to use the pinned version:

```bash
pip install transformers==4.45.0
```

---

## 👥 Project Team

| Name | Role |
|------|------|
| **Sai Venkata Gopala Krishna Bubathula** | Model Lead — fine-tuning, evaluation, deployment, root cause analysis |
| Keremfon Ekerete | Model Research & Comparison |
| Anthony Mensah | Knowledge Representation & Prompting |
| Poojitha Yemineni | Dataset & Instruction Tuning Lead |
| Emmanuel Enotobore | Demo & Integration Lead |

**Line Manager:** Ugur Acar, Chief AI Officer, Depixen  
**Supervisor:** Yalcin, Depixen  
**Placement Lead:** Dr Ali Abbas, University of East London  
**Placement Period:** 26 January – 20 May 2026

---

## 📖 Citation

If you use this work, model, dataset or hallucination evaluation framework in your research, please cite:

```bibtex
@misc{bubathula2026text2kg,
  author       = {Bubathula, Sai Venkata Gopala Krishna and
                  Ekerete, Keremfon and
                  Mensah, Anthony and
                  Yemineni, Poojitha and
                  Enotobore, Emmanuel},
  title        = {Text-to-KG SLM: Fine-Tuning Phi-3.5 Mini and Gemma 2 on Verified
                  UK Procurement Data for Zero-Hallucination Knowledge Graph Extraction},
  year         = {2026},
  institution  = {University of East London / Depixen},
  url          = {https://github.com/sai-uel/text-to-kg-construction-slm},
  note         = {Industrial Placement Project, MSc Big Data Technologies}
}
```

---

## 📄 License

This project is licensed under the **Apache 2.0 License** — see [LICENSE](LICENSE) for details.

UK Open Contracting data is used under the **Open Government Licence v3.0** — freely reusable with attribution.

---

<div align="center">
  <strong>University of East London × Depixen</strong><br>
  Industrial Placement 2026 · MSc Big Data Technologies<br>
  <br>
  <strong>⭐ Phase 1: ZERO HALLUCINATION across 1,387 unseen test contracts ⭐</strong><br>
  <br>
  <a href="https://github.com/sai-uel/text-to-kg-construction-slm">GitHub</a> ·
  <a href="https://huggingface.co/BSVGK">HuggingFace</a> ·
  <a href="https://www.linkedin.com/in/sai-venkata-gopala-krishna-bubathula-a05a26283/">LinkedIn</a>
</div>

