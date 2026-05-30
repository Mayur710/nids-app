# 🛡️ Intelligent Network Intrusion Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7.0-orange?style=for-the-badge&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-0.1-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-ff4b4b?style=for-the-badge&logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4-purple?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLaMA3-yellow?style=for-the-badge)

**An end-to-end AI-powered security system that detects network attacks, explains them in plain English, and answers your security questions.**

[Live Demo](https://nids-app-agbfipydqeekvifyddz6yu.streamlit.app) • [Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)

</div>

---

## What This Project Does

Upload a network traffic CSV file and this system will:

1. **Detect attacks** using a trained Random Forest ML model
2. **Explain each attack** in plain English using LangChain + Groq LLaMA 3
3. **Answer your security questions** via a RAG chatbot backed by OWASP, CVE and Cisco documentation
4. **Show visual analytics** — attack distribution charts, flow statistics and prediction tables

---

##  Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Web App                     │
│                                                          │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │  Upload CSV  │  │  ML Results   │  │  AI Analysis │ │
│  │  & Preview   │  │  & Charts     │  │  & Chatbot   │ │
│  └──────┬───────┘  └───────┬───────┘  └──────┬───────┘ │
└─────────┼──────────────────┼─────────────────┼─────────┘
          │                  │                 │
          ▼                  ▼                 ▼
┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐
│  ML Pipeline    │  │  Visualizer  │  │  GenAI Layer    │
│                 │  │              │  │                  │
│  StandardScaler │  │  Bar Charts  │  │  LangChain +    │
│  PCA (25 comp.) │  │  Metrics     │  │  Groq LLaMA 3   │
│  Random Forest  │  │  Tables      │  │  ChromaDB RAG   │
└─────────────────┘  └──────────────┘  └─────────────────┘
```

---

##  Attack Types Detected

| Attack | Type |
|--------|------|
|  BENIGN | Normal traffic |
|  DDoS | Distributed Denial of Service |
|  DoS Hulk | High-volume HTTP flood |
|  DoS GoldenEye | Slow HTTP DoS |
|  DoS Slowhttptest | Slow HTTP POST |
|  DoS slowloris | Connection exhaustion |
|  PortScan | Network reconnaissance |
|  Bot | Botnet traffic |
|  FTP-Patator | FTP brute force |
|  SSH-Patator | SSH brute force |
|  Web Attack – Brute Force | Web login brute force |
|  Web Attack – XSS | Cross-site scripting |
|  Web Attack – SQL Injection | SQL injection |
|  Infiltration | Network infiltration |
|  Heartbleed | OpenSSL exploit |

---

##  Model Performance

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Decision Tree | 99.9% | 0.74 |
| Random Forest | 99.9% | 0.74 |
| XGBoost | 99.0% | 0.73 |
| Voting Classifier | 99.9% | 0.73 |

> **Final model:** Random Forest — selected for highest macro F1 and best performance on rare attack classes.

---

##  Tech Stack

| Category | Tool |
|----------|------|
| Data Processing | Python, Pandas, NumPy |
| ML Models | Scikit-learn, XGBoost |
| Dimensionality Reduction | PCA |
| Class Balancing | SMOTE |
| GenAI Orchestration | LangChain |
| LLM | Groq LLaMA 3.1 8B Instant |
| Vector Database | ChromaDB |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |

---

##  Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Mayur710/nids-app.git
cd nids-app
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run app.py
```

---

##  Project Structure

```
nids-app/
│
├── app.py                               ← Streamlit web app
├── genai.py                             ← LangChain + RAG chatbot
├── building_vectorstore.py              ← ChromaDB setup
├── model_training.py                    ← ML training pipeline
├── data cleaning and preprocessing.py  ← Phase 1 preprocessing
├── visualization.py                     ← Result visualizations
│
├── security_docs/                       ← RAG knowledge base
│   ├── owasp.txt
│   ├── cve_entries.txt
│   ├── ddos.txt
│   ├── portscan.txt
│   └── botnet.txt
│
├── chroma_db/                           ← Persisted vector store
├── visualizations/                      ← Saved evaluation plots
├── requirements.txt
├── runtime.txt
└── .streamlit/
    └── config.toml
```

---

##  How It Works

### Phase 1 — Data Preprocessing
- Loaded 2.5 million network flow records from CIC-IDS2017
- Cleaned nulls, infinity values, constant columns and encoding issues
- Applied StandardScaler for feature normalization
- Used LabelEncoder for attack type encoding

### Phase 2 — ML Training
- Applied PCA reducing 70 features to 25 components with 95% variance retained
- Handled severe class imbalance using SMOTE with custom per-class strategy
- Trained Decision Tree, Random Forest and XGBoost on stratified 500K sample
- Random Forest selected as final model based on macro F1 evaluation
- Built Voting Classifier combining all three models for comparison

### Phase 3 — GenAI Layer
- Built LangChain prompt template taking attack type, confidence and network features
- Connected to Groq LLaMA 3.1 for fast free inference
- Loaded OWASP, CVE and security documents into ChromaDB vector store
- Built RAG pipeline for context-aware security question answering

### Phase 4 — Deployment
- Built Streamlit app connecting all three phases
- Model files served via Google Drive
- Deployed on Streamlit Cloud with secrets management


## 👤 Author

**Mayur** — [GitHub](https://github.com/Mayur710)

---

##  License

This project is licensed under the MIT License.
