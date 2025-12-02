# 🚀 BERT for Named Entity Recognition (NER) — Fine-Tuning on CoNLL-2003  
**TensorFlow + HuggingFace Transformers**

This repository contains a complete, end-to-end implementation of fine-tuning **BERT (bert-base-cased)** for **Named Entity Recognition (NER)** using the **CoNLL-2003** dataset.  
It includes token–label alignment, dataset preparation, training, model saving, and inference scripts.

---

## 📌 Features
✔️ Fine-tunes `bert-base-cased` for token classification  
✔️ Uses HuggingFace `datasets` + `transformers`  
✔️ Clean label alignment for subword tokens  
✔️ TensorFlow `tf.data` training pipeline  
✔️ Saves model + tokenizer in HF format  
✔️ Easy inference function for custom sentences  

---

## 📂 Project Structure
📁 bert-ner-conll/
│
├── train_ner.py # Main training script
├── README.md # Documentation
└── bert-ner-model/ # Saved model + tokenizer after training

## 📦 Installation

### 1️⃣ Install Dependencies
```
pip install tensorflow transformers datasets numpy
```

### 2️⃣ (Optional) GPU Setup
Ensure your TensorFlow is GPU-enabled if you have CUDA installed.

## 📊 Dataset: CoNLL-2003

This dataset contains annotated tokens for:

PER — Person

ORG — Organization

LOC — Location

MISC — Miscellaneous

## 📈 Results

Model performance depends on training time and GPU availability but BERT typically achieves strong NER accuracy on CoNLL-2003.

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to improve.

## 📝 License

This project is open-source under the MIT License.

## ⭐ Acknowledgements

HuggingFace Transformers

HuggingFace Datasets

TensorFlow

CoNLL-2003 Shared Task Dataset
