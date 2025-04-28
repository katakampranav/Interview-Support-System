# Interview Support System

**Interview Support System** — an AI-powered tool that helps generate **professional interview answers** using a fine-tuned **Falcon-7B** model and **Google Gemini-2.0**.
---

## 🚀 Demo

You can use the app via the **Colab Notebook (`app.ipynb`)**.

```bash
# Open app.ipynb and run all cells (Colab recommended)
```

---

## ✨ Features

- Fine-tuned **Falcon-7B** model for understanding interview questions.
- **Google Gemini 2.0** API for generating detailed, professional answers.
- **Efficient 4-bit quantized model loading** (using BitsandBytes).
- Simple function-based inference using `generate_response()` (see `inference.py`).
- Clean two-step pipeline:
  - **Falcon Model** → Generates **guidance points**.
  - **Gemini Model** → Converts guidance into **final answers**.

---

## 📢 Fine-tuned Model Availability

The **fine-tuned Falcon-7B model** is published and readily available on **Hugging Face Hub**:

👉 **Access it here:**  
[**Falcon-7B QLoRA Interview QA Support Bot**](https://huggingface.co/Pranav06/falcon-7b-qlora-interview_qa-support-bot)

You can directly use it for inference, evaluation, or further fine-tuning without setting up locally.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/katakampranav/Interview-Support-System.git
```

### 2. Install the Requirements
```bash
pip install -r requirements.txt
```

You need:
- Python ≥ 3.8
- A CUDA-compatible GPU (for Falcon model inference)
- Google API Key for Gemini access.

### 3. Running the Fine-tuned Model

**(For Testing Model Loading and Response Generation)**

In `inference.py`, the fine-tuned Falcon-7B model is loaded, and you can test it with:

```python
from inference import generate_response

response = generate_response("Tell me about yourself")
print(response)
```

This will output a **short professional guidance** for the question.

### 4. Full Interview Answer Generation

**Use `app.ipynb`**:  
- It combines **Falcon Guidance** + **Gemini Answer** into a full response.
- A simple Gradio interface is included to interact with both models.

---

## 📂 Project Structure

```
katakampranav-interview-support-system/
│
├── app.ipynb                 # Colab notebook: full app (Falcon + Gemini)
├── finetuning_model.ipynb     # Fine-tuning Falcon-7B model on custom dataset
├── inference.py              # Model loading and basic inference function
├── requirements.txt          # Project dependencies
│
├── Data/
│   └── interview_qa_data.json   # Custom Interview Q&A dataset
│
├── OUTPUT/
│   └── runs/
│       └── ...                  # Fine-tuning logs (TensorBoard files)
│
└── trained-model/
    ├── README.md
    ├── adapter_config.json      # PEFT adapter config
    └── adapter_model.safetensors  # Fine-tuned LoRA model weights
```

---

## 🧪 Model Fine-Tuning Details

The fine-tuning of the **Falcon-7B** model was performed inside the `finetuning_model.ipynb` notebook.

### Fine-Tuning Process:
- **Base Model:** `tiiuae/falcon-7b-instruct`
- **Method:** **QLoRA (Quantized Low-Rank Adaptation)**  
  - Fine-tuning only a small set of adapter parameters.
  - Using 4-bit quantization for memory efficiency (BitsAndBytes).
- **Library:** Hugging Face Transformers + PEFT
- **Dataset:** `Data/interview_qa_data.json`  
  - Contains a custom dataset of **interview questions and answers**.
- **Training Setup:**
  - Optimizer: AdamW
  - Scheduler: Cosine learning rate scheduler
  - Mixed precision: bfloat16
  - Training logs saved to `OUTPUT/runs/` (for TensorBoard)'
  - And also pushed to hugginfface hub

At the end of fine-tuning:
- The **adapter weights** are saved inside `trained-model/`:
  - `adapter_model.safetensors`
  - `adapter_config.json`
  
✅ This allows **efficient loading** of the fine-tuned model without retraining.

---

## ⚙️ How It Works

- **Step 1:** The user submits an interview question.
- **Step 2:** The fine-tuned **Falcon-7B** model provides guidance points on how to answer the question.
- **Step 3:** **Google Gemini** uses the guidance to generate a fully polished, first-person interview answer.
- **Step 4:** The system displays both outputs.

---

## Example Outputs
### Example 1
![Image](https://github.com/user-attachments/assets/cd566d73-f2ae-4529-9bb7-7aef79b70b79)
![Image](https://github.com/user-attachments/assets/7b409b6f-48a5-49d3-b44d-3e5b0e3e6fb9)

### Example 2
![Image](https://github.com/user-attachments/assets/2da6d504-883a-48cb-904d-e5bbc78a8982)
![Image](https://github.com/user-attachments/assets/0b1d0228-6992-4feb-9b63-3c9a609240b0)

---

## 🛠️ Tech Stack

- **Python**
- **Gradio** (UI)
- **HuggingFace Transformers** (Falcon model)
- **PEFT** (Parameter Efficient Fine-tuning)
- **BitsAndBytes** (4-bit quantization)
- **Google Generative AI API** (Gemini)

---

## 📌 Future Work

- Add model selection options (multiple LLM outputs).
- Deploy to Hugging Face Spaces / AWS EC2.
- Enhance UX: Save generated answers, Email export, etc.

---

## 📜 Credits

Built by **[Katakam Pranav Shankar](https://github.com/katakampranav)**  
Fine-tuning and system design powered by Hugging Face, Gradio, and Google AI.

---
