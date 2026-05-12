# Text to Knowledge Graph Extraction

## Fine-Tuning SLMs with LoRA and QLoRA on UK Construction Contracts

This project investigates Text to Knowledge Graph (Text-to-KG) extraction using Small Language Models (SLMs). The system converts unstructured UK public procurement contract descriptions into structured RDF knowledge graph triples.

The project was developed as part of the UEL – Depixen collaboration exploring domain-aligned AI systems trained on verified data sources.

---

## Project Overview

Construction contract descriptions contain rich structured information that is locked inside unstructured text. This project focuses on automatically extracting this information as structured knowledge graph triples using fine-tuned language models.

The project pipeline includes:

- Domain selection
- Dataset extraction and preparation
- Ontology design
- Text-to-KG dataset creation
- Fine-tuning SLMs using LoRA and QLoRA
- Evaluation using multiple metrics
- Model deployment and inference interface

The final system takes a UK construction contract description as input and generates structured RDF triples ready for knowledge graph ingestion.

---

## Project Overview

**Domain:** UK Public Procurement / Construction Contracts
**Datasource Link:** https://data.open-contracting.org/en/

The UK construction procurement domain was selected because it contains highly structured award information published by public authorities, making it ideal for ontology modeling and knowledge graph extraction.

Reasons for selecting this domain:

- Strong alignment with verified open government data sources
- High importance of factual accuracy in legal and financial contexts
- Ideal for ontology design and semantic relationships
- Well suited for Text to Knowledge Graph extraction
- Aligns with Depixen's focus on trustworthy AI systems

---

## Dataset

**Dataset:** BSVGK/Text_to_KG_Construction_Dataset

| Field | Description |
|---|---|
| Contract ID | Unique identifier for each contract record |
| Buyer | The organisation awarding the contract |
| Supplier | The organisation receiving the contract |
| Contract Value | Monetary value of the contract in GBP |
| Award Date | Date the contract was awarded |
| CPV Code | Common Procurement Vocabulary classification code |
| CPV Description | Text description of the CPV category |
| Location | Geographic region of the contract |

**Dataset Repository:** https://huggingface.co/datasets/BSVGK/Text_to_KG_Construction_Dataset

---

## Dataset Preparation Pipeline

The dataset preparation process consisted of the following stages.

**1. Source Data Collection**

UK public procurement contract records were collected and processed into structured format.

**2. Ontology Design**

An RDF ontology was designed to represent contract entities and their properties.

**3. Triple Generation**

Structured triples were generated from the contract records following the ontology schema.

**4. Instruction Dataset Creation**

Triples were transformed into instruction-based training samples. Each sample includes Instruction, Input and Output.

**5. Train / Validation / Test Split**

| Split | Number of Samples |
|---|---|
| Train | 6,470 |
| Validation | 1,387 |
| Test | 1,387 |
| **Total** | **9,244** |

---

## Ontology Design

An RDF ontology was designed to represent construction contract entities and their properties.

**Classes (5):**

| Class | Description |
|---|---|
| Contract | A public construction contract record |
| Buyer | The organisation purchasing or awarding the contract |
| Supplier | The organisation awarded the contract |
| CPVCategory | The procurement classification category |
| Location | The geographic region of the contract |

**Relations (8):**

| Predicate | Meaning |
|---|---|
| rdf:type | Entity class assignment |
| hasBuyer | Links contract to the buying organisation |
| hasSupplier | Links contract to the awarded supplier |
| hasContractValue | Monetary value of the contract |
| hasAwardDate | Date the contract was awarded |
| hasCPVCode | Procurement classification code |
| hasCPVDescription | Description of the CPV category |
| hasLocation | Geographic region of the contract |

**Triple Format:**

```
(Contract_008672, rdf:type, Contract)
(Contract_008672, hasBuyer, Wakefield Council Customer Service)
(Contract_008672, hasSupplier, AMALGAMATED CONSTRUCTION LTD)
(Contract_008672, hasContractValue, 420850)
(Contract_008672, hasAwardDate, 2017-12-05)
(Contract_008672, hasCPVCode, 71322000)
(Contract_008672, hasCPVDescription, Engineering design services for the construction of civil engineering works)
(Contract_008672, hasLocation, Yorkshire and the Humber)
```

---

## Model Selection

Two base models were evaluated.

**Phase 1:** microsoft/Phi-3.5-mini-instruct

Phi-3.5 Mini is a compact instruction-tuned language model developed by Microsoft.

Reasons for selecting Phi-3.5 Mini:
- Strong instruction-following ability
- Efficient small language model at 3.82B parameters
- Suitable for parameter-efficient fine-tuning with LoRA
- Good balance between performance and compute cost

**Phase 2:** google/gemma-2-2b-it

Gemma 2 2B is a lightweight instruction-tuned language model developed by Google.

Reasons for selecting Gemma 2 2B:
- Highly memory efficient at 2.3 GB with 4-bit quantization
- Strong structured output generation
- Suitable for QLoRA fine-tuning on limited hardware
- Direct comparison with Phase 1 for model size vs quality analysis

---

## Fine-Tuning Methods

**LoRA (Low-Rank Adaptation) — Phase 1**

LoRA freezes the base model weights and trains small low-rank matrices added to specific attention and feed-forward layers.

Advantages:
- Efficient training with low memory requirements
- Stable optimization behavior
- Small adapter size (34 MB vs 7.64 GB full model)

| Parameter | Value |
|---|---|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | 8,912,896 |
| Total parameters | 3,829,992,448 |
| Trainable percentage | 0.2327% |

**QLoRA (Quantized LoRA) — Phase 2**

QLoRA loads the base model using 4-bit NF4 quantization and trains LoRA adapters on the quantized weights.

Advantages:
- Significantly lower GPU memory usage (2.3 GB vs 7.64 GB)
- Enables larger model training on memory-constrained hardware
- Strong performance despite quantization

| Parameter | Value |
|---|---|
| Quantization | 4-bit NF4 |
| Compute dtype | bfloat16 |
| Double quantization | True |
| Rank (r) | 16 |
| Alpha | 32 |
| Trainable parameters | 20,766,720 |
| Trainable percentage | 1.2795% |

---

## Training Configuration

| Parameter | Phase 1 (LoRA) | Phase 2 (QLoRA) |
|---|---|---|
| Base model | Phi-3.5 Mini | Gemma 2 2B IT |
| Epochs | 3 | 3 |
| Batch size | 4 | 4 |
| Gradient accumulation | 4 | 4 |
| Effective batch size | 16 | 16 |
| Learning rate | 2e-4 | 2e-4 |
| Scheduler | Cosine | Cosine |
| Optimizer | AdamW | Paged AdamW 8-bit |
| Hardware | NVIDIA A100 80GB | NVIDIA A100 40GB |
| Training steps | 1,212 | 1,212 |
| Training time | 26.01 minutes | 62.37 minutes |

---

## Evaluation Metrics

Model performance was evaluated using multiple metrics.

**Triple Extraction Metrics**
- F1 Score
- Precision
- Recall

**Text Similarity Metrics**
- ROUGE-L
- BERTScore

**Hallucination Metrics**
- Hallucination L1 — proportion of predicted triples with invalid relation types not in the ontology
- Hallucination L2 — proportion of predicted triples with entity values not grounded in the input text

---

## Results

**Phase 1 — Phi-3.5 Mini Instruct + LoRA**

| Metric | Score |
|---|---|
| F1 Score | 0.9954 |
| Recall | 0.9950 |
| Precision | 0.9959 |
| ROUGE-L | 0.9991 |
| BERTScore | 0.9997 |
| Hallucination L1 | 0.0 |
| Hallucination L2 | 0.0 |
| Training Loss | 0.3008 |
| Validation Loss | 0.2570 |
| Training Time | 26.01 minutes |

**Phase 2 — Gemma 2 2B IT + QLoRA**

| Metric | Score |
|---|---|
| F1 Score | 0.9756 |
| Recall | 0.9688 |
| Precision | 0.9835 |
| ROUGE-L | 0.9783 |
| BERTScore | 0.9975 |
| Hallucination L1 | 0.0036 |
| Hallucination L2 | 0.1233 |
| Training Loss | 0.2689 |
| Validation Loss | 0.2563 |
| Training Time | 62.37 minutes |

---

## Model Comparison

| Metric | Phi-3.5 + LoRA | Gemma 2 + QLoRA | Winner |
|---|---|---|---|
| F1 Score | 0.9954 | 0.9756 | Phi-3.5 + LoRA |
| Recall | 0.9950 | 0.9688 | Phi-3.5 + LoRA |
| Precision | 0.9959 | 0.9835 | Phi-3.5 + LoRA |
| ROUGE-L | 0.9991 | 0.9783 | Phi-3.5 + LoRA |
| BERTScore | 0.9997 | 0.9975 | Phi-3.5 + LoRA |
| Hallucination L1 | **0.0** | 0.0036 | Phi-3.5 + LoRA |
| Hallucination L2 | **0.0** | 0.1233 | Phi-3.5 + LoRA |
| Training Time | 26.01 min | 62.37 min | Phi-3.5 + LoRA |
| Memory Used | 7.64 GB | 2.3 GB | Gemma 2 + QLoRA |
| Model Size | 3.82B | 2.6B | Gemma 2 + QLoRA |
| **Weighted Score** | **0.8469** | **0.8250** | **Phi-3.5 + LoRA** |

**Observations:**

- Phi-3.5 Mini + LoRA achieved near-perfect triple extraction with zero hallucination on both L1 and L2 across all 1,387 test records
- Gemma 2 + QLoRA is significantly more memory efficient at 2.3 GB vs 7.64 GB making it suitable for resource-constrained environments
- Phi-3.5 LoRA converged in 26 minutes vs 62 minutes for Gemma 2 QLoRA despite being a larger model
- Both models achieved strong results above 0.97 F1 confirming that parameter-efficient fine-tuning of SLMs is highly effective for structured KG extraction

**Final preferred model: Phi-3.5 Mini Instruct + LoRA**

Selected based on highest weighted score (0.8469), perfect hallucination score, and fastest training time.

---

## HuggingFace Repositories

**Dataset Repository**

https://huggingface.co/datasets/BSVGK/Text_to_KG_Construction_Dataset

Contains:
- Text-to-KG dataset
- Train / validation / test splits
- Instruction formatted samples

**LoRA Adapter Model**

https://huggingface.co/BSVGK/phi35-mini-lora-text2kg-adapter

**Merged Model (Recommended for inference)**

https://huggingface.co/BSVGK/phi35-mini-lora-text2kg-merged

---

## Inference

**Loading the merged model:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

REPO_ID = "BSVGK/phi35-mini-lora-text2kg-merged"

tokenizer = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(
    REPO_ID,
    torch_dtype         = torch.float16,
    device_map          = "auto",
    trust_remote_code   = True,
    attn_implementation = "eager"
)

def extract_triples(contract_text):
    prompt = f"""<|user|>
Extract entities and relationships from the construction contract description and represent them as triples.

Contract Description:
{contract_text}<|end|>
<|assistant|>
"""
    inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
```

---

## Interface Implementation

A Gradio inference interface was built for interactive model testing.

**Features:**
- Free text input for contract descriptions
- File upload support (CSV and PDF)
- Load uploaded file into text box
- Download format selector (CSV or TXT)
- Generate Output button
- Download predictions as file
- Session history tab

---

## GitHub Repository

https://github.com/sai-uel/text-to-kg-construction-slm

**Notebooks:**

| Notebook | Description |
|---|---|
| Phi3.5MiniInstruct_LoRA.ipynb | Phase 1 training, evaluation and deployment |
| Gemma2_2b_it_QLoRA.ipynb | Phase 2 training, evaluation and deployment |
| Comparison_Phi_Gemma.ipynb | Model comparison, visualizations and best model selection |
| Gradio_Interface.ipynb | Inference interface for the deployed model |

---

## Limitations

- The model was trained on a fixed ontology of 8 relation types and cannot extract relations outside this schema
- The dataset consists of structured and repetitive UK procurement records. Real-world contracts with greater linguistic variation may show lower performance
- Inference requires transformers version 4.45.0 due to compatibility issues with newer versions
- The model was trained exclusively on English UK public procurement contracts and has not been validated on other domains or languages
- Evaluation relies entirely on automated metrics. Human expert evaluation was not conducted

---

## Future Work

- Extend the ontology to support additional relation types and entity classes
- Train on contracts from multiple countries and legal systems to improve generalization
- Introduce data augmentation to increase dataset diversity
- Evaluate on real-world contract documents with more complex and varied language
- Deploy as a persistent API service for continuous use without session expiry

---

## Acknowledgements

University of East London

Depixen Collaboration

HuggingFace

Microsoft Phi-3.5

Google Gemma
