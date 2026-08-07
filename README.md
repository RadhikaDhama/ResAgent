# 🚀 ResAgent: Agentic AI Portfolio & Candidate Match Copilot

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Python 3.11](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FAISS Vector Store](https://img.shields.io/badge/FAISS-Vector%20Search-0467DF?style=for-the-badge)

**ResAgent** is an interactive, multi-tool Agentic AI Web Application built with **LangChain**, **Google Gemini API**, **FAISS Vector RAG**, and **Streamlit**. It represents **Radhika Dhama** (M.Sc. Data Science @ Chennai Mathematical Institute), enabling recruiters and technical managers to conversationally explore technical qualifications, evaluate job description alignment, inspect code snippets, and view live GitHub repositories.

---

## 🌟 Key Architecture & Features

### 1. 🤖 Multi-Tool Agentic AI Copilot
* Built using **LangChain Tool-Calling Agents** and **FAISS Vector Search**.
* Answers questions about speech model fine-tuning (Orpheus-3B), statistical out-of-distribution (OOD) detection (GMM + Mahalanobis), quantitative portfolio SLSQP solvers, and academic ranks.

### 2. 🎛️ Recruiter Role Perspective Switching
* Switch between 4 target candidate profiles:
  1. **Graduate Engineer — AI & Full Stack (HP Inc)**
  2. **Speech & Generative AI ML Engineer**
  3. **Quantitative Analyst & Financial Risk**
  4. **Data Scientist & Statistical Modeler**

### 3. 🎯 Transparent Job Description (JD) Alignment Studio
* Computes an **Overall Candidate Match Score** with a transparent mathematical derivation:
$$\text{Match Score} = (0.40 \times \text{Skill Alignment}) + (0.30 \times \text{Academic Quality}) + (0.30 \times \text{Industry Experience})$$

### 4. 🐙 Live GitHub API Repository Integration
* Dynamically syncs public repositories live from `github.com/RadhikaDhama` using GitHub's REST API, automatically filtering projects by selected target role.

### 5. 📊 Executive Skill Matrix Visualization
* Interactive horizontal bar chart and polar radar chart mapping core competencies across deep learning, statistics, speech processing, and full-stack web development.

---

## 📁 Repository Structure

```text
ResAgent/
├── app.py                      # Main Streamlit Dashboard UI & Layout
├── agent_engine.py             # LangChain Tool Engine & Gemini Agent Logic
├── requirements.txt            # Project Dependencies
├── README.md                   # Project Documentation
└── data/
    ├── resume_knowledge.txt    # Vector Store Source Knowledge Base
    └── agentic-ai-prototype.ipynb # Notebook for Kaggle / Colab Deployment
```

---

## 💻 Local Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/RadhikaDhama/ResAgent.git
   cd ResAgent
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Streamlit Application:**
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

1. Fork or push this repository to your GitHub account (`RadhikaDhama/ResAgent`).
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **New app**, select `RadhikaDhama/ResAgent`, set Repository: `RadhikaDhama/ResAgent`, Branch: `main`, Main file path: `app.py`.
4. Click **Deploy!** Your app will be live on an HTTPS link (e.g. `https://resagent.streamlit.app`).

---

## 📄 Candidate Profile Summary

* **Candidate:** Radhika Dhama
* **Education:** M.Sc. Data Science @ Chennai Mathematical Institute (CGPA: 8.69) | B.Sc. Statistics Hons @ DU (College Rank 1, CGPA: 9.33)
* **Honors:** IIT JAM Mathematical Statistics AIR 66 (2025) | GATE Statistics AIR 142 (2025) | Accenture Strategy Connect S4 PPI Recipient
* **LinkedIn:** [linkedin.com/in/radhika-dhama](https://www.linkedin.com/in/radhika-dhama/)
* **GitHub:** [github.com/RadhikaDhama](https://github.com/RadhikaDhama)
