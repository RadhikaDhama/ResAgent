# 🚀 ResAgent: Agentic AI Portfolio & Candidate Match Copilot

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://resagent-5cuarzs4hakuzhmpz7ygrl.streamlit.app/)
![LangChain RAG](https://img.shields.io/badge/LangChain-RAG%20Agent-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Python 3.11](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FAISS Vector Store](https://img.shields.io/badge/FAISS-Vector%20Search-0467DF?style=for-the-badge)

🔗 **Live Interactive Application:** [https://resagent-5cuarzs4hakuzhmpz7ygrl.streamlit.app/](https://resagent-5cuarzs4hakuzhmpz7ygrl.streamlit.app/)

**ResAgent** is an interactive, multi-tool Agentic AI Web Application powered by **Retrieval-Augmented Generation (RAG)**, **LangChain**, **Google Gemini API**, **FAISS Vector Search**, and **Streamlit**. It represents **Radhika Dhama** (M.Sc. Data Science @ Chennai Mathematical Institute), enabling recruiters and technical hiring managers to conversationally explore technical qualifications, evaluate job description alignment, inspect code snippets, and view live GitHub repositories.

---

## 🌟 Key Architecture & Features

### 1. RAG-Powered Agentic AI Copilot (Retrieval-Augmented Generation)
* Built using **LangChain Tool-Calling Agents** and **FAISS Vector RAG**.
* Dynamically retrieves relevant context from a curated vector database to answer queries about speech LLM fine-tuning (Orpheus-3B), statistical out-of-distribution (OOD) detection (GMM + Mahalanobis), quantitative portfolio SLSQP solvers, and academic ranks.

### 2. Recruiter Role Perspective Switching
* Switch between 4 target candidate profiles:
  1. **Graduate Engineer — AI & Full Stack (HP Inc)**
  2. **Speech & Generative AI ML Engineer**
  3. **Quantitative Analyst & Financial Risk**
  4. **Data Scientist & Statistical Modeler**

### 3. Transparent Job Description (JD) Alignment Studio
* Evaluates job descriptions with exact word-boundary technology matching and a strict domain relevance gate to output clear match reports and scores.

### 4. Live GitHub API Repository Integration
* Dynamically syncs public repositories live from `github.com/RadhikaDhama` using GitHub's REST API, automatically filtering projects by selected target role.

### 5. Executive Skill Matrix Visualization
* Interactive horizontal bar chart and polar radar chart mapping core competencies across deep learning, statistics, speech processing, and full-stack web development.

---

## 📁 Repository Structure

```text
ResAgent/
├── app.py                      # Main Streamlit Dashboard UI & Layout
├── agent_engine.py             # LangChain RAG Tool Engine & Gemini Agent Logic
├── requirements.txt            # Project Dependencies
├── README.md                   # Project Documentation
└── data/
    ├── resume_knowledge.txt    # FAISS Vector Store Knowledge Base
    └── agentic-ai-prototype.ipynb # Notebook for Kaggle / Colab Deployment
```

---

## 🌐 Live Web Application

Experience the live deployed application on Streamlit Cloud:
 **[https://resagent-5cuarzs4hakuzhmpz7ygrl.streamlit.app/](https://resagent-5cuarzs4hakuzhmpz7ygrl.streamlit.app/)**

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

## 📄 Candidate Profile Summary

* **Candidate:** Radhika Dhama
* **Education:** M.Sc. Data Science @ Chennai Mathematical Institute (CGPA: 8.69) | B.Sc. Statistics Hons @ DU (College Rank 1, CGPA: 9.33)
* **Honors:** IIT JAM Mathematical Statistics AIR 66 (2025) | GATE Statistics AIR 142 (2025) | Accenture Strategy Connect S4 PPI Recipient
* **LinkedIn:** [linkedin.com/in/radhika-dhama](https://www.linkedin.com/in/radhika-dhama/)
* **GitHub:** [github.com/RadhikaDhama](https://github.com/RadhikaDhama)
* **Live App:** [resagent-5cuarzs4hakuzhmpz7ygrl.streamlit.app](https://resagent-5cuarzs4hakuzhmpz7ygrl.streamlit.app/)
