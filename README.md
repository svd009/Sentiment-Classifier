# Sentiment Classifier (Fine-Tuned BERT)

A FastAPI-based sentiment analysis API built using a fine-tuned **BERT** model.  
The model is trained on the IMDB dataset and can classify text as **Positive** or **Negative**.

---

## **Features**
- Fine-tuned `bert-base-uncased` model using Hugging Face Transformers.
- REST API built with **FastAPI**.
- Predicts sentiment with confidence scores.
- Easy to deploy locally or on cloud (e.g. AWS, Render, Railway).

---

## **Tech Stack**
- **Python 3.10+**
- **Hugging Face Transformers** & **Datasets**
- **PyTorch**
- **FastAPI** + **Uvicorn**

---

## **Setup Instructions**

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/svd009/Sentiment-Classifier.git
cd Sentiment-Classifier
