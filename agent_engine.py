import os
import re
import base64
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

_VECTORSTORE = None
_B64_KEY = "QVEuQWI4Uk42SUE2WGFCRUpRMGZXWXN4YmdWT2V3aHBOeUtDMUxsdWpRb3VUMHhGWkp4WXc="

def get_default_api_key():
    env_key = os.getenv("GOOGLE_API_KEY", "")
    if env_key:
        return env_key
    try:
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    try:
        return base64.b64decode(_B64_KEY).decode("utf-8")
    except Exception:
        return ""

def get_or_build_vectorstore():
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "resume_knowledge.txt")
    if not os.path.exists(data_path):
        data_path = "resume_knowledge.txt"
        
    loader = TextLoader(data_path, encoding="utf-8")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=60)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    _VECTORSTORE = FAISS.from_documents(chunks, embeddings)
    return _VECTORSTORE

@tool
def search_candidate_portfolio(query: str) -> str:
    """Searches Radhika Dhama's resume knowledge base to retrieve accurate, context-specific information for any query."""
    try:
        vs = get_or_build_vectorstore()
        docs = vs.similarity_search(query, k=4)
        return "\n\n---\n\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"Error retrieving context from vector database: {str(e)}"

@tool
def analyze_job_description_fit(job_description: str) -> str:
    """Evaluates how well Radhika Dhama fits a given Job Description (JD) text provided by a recruiter with a transparent score breakdown."""
    candidate_skills = [
        "Python", "PyTorch", "Machine Learning", "Statistics", "LangChain", 
        "LLMs", "Streamlit", "GMM", "SQL", "Git", "Plotly", "LoRA", 
        "Unsloth", "Whisper", "wav2vec2", "RAG", "OOD", "ASR", 
        "Deep Learning", "Data Science", "R"
    ]
    jd_lower = job_description.lower() if job_description else ""
    
    # Exact word boundary matching (prevents false positives like 'r' matching 'pastry')
    matched = []
    for skill in candidate_skills:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, jd_lower):
            matched.append(skill)
    
    academic_score = 98
    experience_score = 92
    
    if not matched:
        skill_score = 0
        overall_fit_score = 0
        verdict = "**Domain Mismatch:** This job description does not match Radhika Dhama's core domain (Data Science, AI, Speech LLMs, Quantitative Risk)."
    else:
        skill_score = min(100, int((len(matched) / 4.0) * 100))
        overall_fit_score = int((skill_score / 100.0) * (0.50 * skill_score + 0.25 * academic_score + 0.25 * experience_score))
        if overall_fit_score >= 80:
            verdict = "**Strong Match:** Excellent candidate fit for AI Engineering, Machine Learning, Speech/LLM, Data Science, or Quantitative Analyst roles."
        elif overall_fit_score >= 50:
            verdict = "**Moderate Match:** Partial technology stack match with strong core academic analytical skills."
        else:
            verdict = "**Low Match:** Limited overlap with target candidate profile."

    return f"""
### Candidate Fit & Match Score Report — Radhika Dhama

**Overall Candidate Match Score: {overall_fit_score} / 100**

---

#### Score Calculation Weighting:
`50% Skill Alignment` | `25% Academic Quality` | `25% Industry Experience`

1. **Skill Keyword Alignment (50% Weight): {skill_score}%**
   - **Matched Core Technologies:** {', '.join([f'`{s}`' for s in matched]) if matched else 'None (No relevant technology keywords found in Job Description)'}
   - **Assessment:** Found {len(matched)} direct matching tech-stack keywords from target job description.

2. **Academic & Research Rigor (25% Weight): {academic_score if matched else 0}%**
   - **M.Sc. in Data Science (CMI):** Advanced Machine Learning, Algorithm Design, Linear Algebra. (CGPA: 8.69)
   - **B.Sc. Statistics Hons (DU):** College Rank 1 (9.33 CGPA), IIT JAM AIR 66, GATE Stat AIR 142.

3. **Practical Industry & Internship Experience (25% Weight): {experience_score if matched else 0}%**
   - **Speech Fine-Tuning & LLMs:** Cut WER by 66.6% (0.92 -> 0.31) on Hindi TTS via 3-stage LoRA pipeline at Coriolis Technologies.
   - **Statistical ML & Security:** Designed per-class GMM + Relative Mahalanobis OOD detection with 99.59% accuracy across 41 classes.

---

**Verdict:** {verdict}
"""

@tool
def get_project_code_snippet(topic: str) -> str:
    """Returns technical architecture, mathematical formulations, and Python code snippets for Radhika's projects."""
    t_lower = topic.lower()
    if "customer" in t_lower or "personality" in t_lower or "dashboard" in t_lower:
        return """
### Technical Snippet: Customer Personality Analysis Dashboard (Streamlit & Plotly)
**GitHub Repository:** https://github.com/RadhikaDhama/Customer-Personality-Analysis-Dashboard

```python
import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def load_customer_data():
    df = pd.read_csv("marketing_campaign.csv", sep="\t")
    df['Age'] = 2025 - df['Year_Birth']
    df['Total_Spending'] = df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
    return df

df = load_customer_data()
st.title("Customer Personality Analysis Dashboard")

fig = px.scatter(df, x="Income", y="Total_Spending", color="Education",
                 size="Age", hover_name="ID", title="Income vs Total Spending by Education")
st.plotly_chart(fig, use_container_width=True)
```
"""
    elif "ood" in t_lower or "mahalanobis" in t_lower or "gmm" in t_lower:
        return r"""
### Technical Snippet: Statistical OOD Detection (GMM + Relative Mahalanobis Distance)
```python
import numpy as np
from sklearn.mixture import GaussianMixture
class StatisticalOODDetector:
    def __init__(self, n_classes=41):
        self.gmm_models = {}
```
"""
    elif "lora" in t_lower or "tts" in t_lower or "speech" in t_lower:
        return """
### Technical Snippet: 3-Stage LoRA Fine-Tuning Pipeline (Hindi TTS Orpheus-3B)
"""
    else:
        return """
### Technical Snippet: Mean-Variance Portfolio Optimization & VaR
"""

def extract_clean_text(ans_content) -> str:
    if isinstance(ans_content, list):
        parts = []
        for item in ans_content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts) if parts else str(ans_content)
    return str(ans_content)

class ResAgentEngine:
    def __init__(self, api_key: str = None):
        if not api_key:
            api_key = get_default_api_key()
        model_candidates = ["gemini-flash-latest", "gemini-2.0-flash-lite", "gemini-3.6-flash", "gemini-2.0-flash", "gemini-pro"]
        self.llm = None
        for model_name in model_candidates:
            try:
                self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.2)
                break
            except Exception:
                continue
        if self.llm is None:
            self.llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=api_key, temperature=0.2)
            
        self.tools = {
            "search_candidate_portfolio": search_candidate_portfolio,
            "analyze_job_description_fit": analyze_job_description_fit,
            "get_project_code_snippet": get_project_code_snippet
        }
        self.llm_with_tools = self.llm.bind_tools(list(self.tools.values()))

    def invoke(self, inputs: dict) -> dict:
        user_text = inputs.get("input", "")
        system_prompt = "You are ResAgent, an executive AI copilot for Radhika Dhama. Answer any question about her resume, education, projects, job fit, or technical depth directly and concisely."
        full_query = f"{system_prompt}\n\nUser Question: {user_text}"
        try:
            ai_msg = self.llm_with_tools.invoke(full_query)
            if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                tool_outputs = []
                for tool_call in ai_msg.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    if tool_name in self.tools:
                        tool_func = self.tools[tool_name]
                        arg_val = list(tool_args.values())[0] if tool_args else user_text
                        tool_res = tool_func.invoke(arg_val)
                        tool_outputs.append(f"{tool_res}")
                context_str = "\n\n".join(tool_outputs)
                final_prompt = f"User Question: {user_text}\n\nRetrieved Data:\n{context_str}\n\nProvide a concise, tailored Markdown response addressing the question directly:"
                final_ans = self.llm.invoke(final_prompt)
                return {"output": extract_clean_text(final_ans.content)}
            else:
                return {"output": extract_clean_text(ai_msg.content)}
        except Exception as e:
            vs_docs = search_candidate_portfolio.invoke(user_text)
            return {"output": f"### Candidate Information — Radhika Dhama\n\n{vs_docs}"}

def get_resagent_executor(api_key: str = None):
    return ResAgentEngine(api_key or get_default_api_key())
