# Intent-Classification

A compact intent classification project that provides an end-to-end pipeline for:
- preparing a vector database (Chroma),
- setting up the LLM environment,
- running inference to classify intents.

## Dataset Description

This project uses a **custom, large-scale e-commerce intent classification dataset** designed to reflect realistic retail and online-shopping customer interactions. The dataset covers **98 granular e-commerce intents**, each with **~900 high-quality utterances**, resulting in a corpus of **~90,000** labeled examples.

### Dataset Composition

Each record contains:
- **intent** — canonical intent name (string)
- **utterance** — user’s free-form message expressing the intent (string)

Total size:
- **98 intents**
- **~900 utterances per intent**
- **~90,000 labeled examples**

### Source Breakdown

The dataset is composed of two parts:

#### **1. Real Human Utterances (47 intents)**
The set of utterances taken from the open-source dataset: **`bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset`** (HuggingFace)

#### **2. Synthetic Utterance Generation (51 intents)**

The synthetic utterances were generated using a **controlled LLM-based synthetic data pipeline** using **Groq – GPT-OSS-20B**

The generation system enforced:
- **No PII**  
- **No real brand/store names**  
- **Strict semantic alignment** with intent description  
- **Tone variety** (neutral, polite, frustrated, casual, urgent)  
- **Channel diversity** (app, web, store, chat, phone)  
- **Short/medium-length conversational utterances**  
- **Mild noise injection** (typos/slang ≤10%)  
- **Distinctness constraints** to avoid duplicates

This ensures that the synthetic utterances maintain **human-like linguistic variability**, while remaining **faithful** to the intent’s semantic definition.

Note: The generation code and prompts are present in `Notebook/Data-Curation.ipynb`.

---

### Intent Coverage

The dataset includes intents across the full e-commerce lifecycle:

- **Cart & Wishlist Actions**  
  (add_product, remove_product, add_to_wishlist, check_cart_items, …)

- **Order Placement & Checkout**  
  (checkout_guest, apply_discount_code, select_payment_plan, …)

- **Delivery & Fulfillment**  
  (delivery_issue, track_delivery, schedule_delivery, damaged_delivery, …)

- **Returns, Refunds & Exchanges**  
  (return_product, exchange_product, refund_status, cancel_order, …)

- **Account & Profile Management**  
  (recover_password, open_account, close_account, change_account, …)

- **Product Discovery & Metadata**  
  (product_information, check_product_reviews, availability, price_match_request, …)

- **Customer Service & Escalation**  
  (human_agent, customer_service, submit_feedback, …)

- **Loyalty, Rewards & Payments**  
  (redeem_gift_card, loyalty_points_balance_query, payment_issue, …)

---

## Quick start (recommended)

These instructions assume you are on a Unix-like machine (Linux or macOS).  
Adjust shell commands for Windows (PowerShell / conda on Windows).

---

### 1. Clone the repository

```bash
git clone https://github.com/RahulNenavath/Intent-Classification.git
cd Intent-Classification
```

### 2. Create a conda environment
If you don’t have Miniconda installed, use the helper script:
```bash
chmod +x setup_miniconda.sh
./setup_miniconda.sh
```
Then create and configure the project environment:
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Install Python dependencies and the package (editable)
From the repository root:
```bash
pip install -e .
```
This installs all dependencies (from pyproject.toml + requirements.txt) and registers the intent_classification package in editable development mode.

### 4. Move to src directory
The Python package lives inside src/:
```bash
cd src
```

### 5. Set up the Chroma vector DB
This initializes and populates your persistent Chroma database using your utterance dataset + intent descriptions.
From the project root:
```bash
python -m vector_db.setup_db
```
This will:
- create/connect to the persistent Chroma DB directory (configured in src/config.py)
- embed utterances
- compute representative utterances per intent (k-center greedy)
- upsert into Chroma collections

### 6. Set up the LLM runtime
The project includes a setup script for the local LLM backend MLX:
```bash
python -m llm.setup_llm
```
Run the above script to download the model from `mlx-community` on Huggingface, and copies model files into `model` folder. Checkout the src/config.py for model name and settings.

### 7. Run inference (RAG-based intent classification)
Execute the main inference module:
```bash
python -m inference
```
This will:
- accept a user utterance
- retrieve candidate intents + representative utterances from Chroma
- pass them into the LLM for RAG-based classification
- output the predicted intent
