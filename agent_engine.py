import os
import re
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

_VECTORSTORE = None

def get_default_api_key():
    """Read API key from environment variable or Streamlit secrets only."""
    env_key = os.getenv("GOOGLE_API_KEY", "")
    if env_key:
        return env_key
    try:
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
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
    """Evaluates how well Radhika Dhama fits a given Job Description (JD) using transparent skill coverage and RAG-grounded qualitative analysis."""
    
    # Radhika's actual skills (what she has)
    candidate_skills = {
        "Python", "PyTorch", "Machine Learning", "Statistics", "LangChain", 
        "LLMs", "Streamlit", "GMM", "SQL", "Git", "Plotly", "LoRA", 
        "Unsloth", "Whisper", "wav2vec2", "RAG", "OOD", "ASR", 
        "Deep Learning", "Data Science", "R", "Pandas", "NumPy",
        "Scikit-learn", "XGBoost", "LightGBM", "SMOTE", "LaTeX",
        "Hugging Face", "FAISS", "SciPy", "EWMA", "Streamlit Cloud"
    }
    
    # Broader vocabulary — includes skills Radhika does NOT have
    # so coverage can realistically be < 100%
    jd_vocabulary = candidate_skills | {
        "React", "JavaScript", "TypeScript", "Node.js", "Docker", "Kubernetes",
        "AWS", "GCP", "Azure", "Java", "C++", "Go", "Rust", "Scala", "Spark",
        "Kafka", "Airflow", "MLflow", "Terraform", "CI/CD", "REST API",
        "MongoDB", "PostgreSQL", "Redis", "GraphQL", "TensorFlow", "Keras",
        "Computer Vision", "NLP", "Tableau", "Power BI", "Excel", "MATLAB",
        "Hadoop", "Hive", "Databricks", "Flask", "Django", "FastAPI",
        "HTML", "CSS", "Vue", "Angular", "Spring Boot", "Microservices"
    }
    
    jd_lower = job_description.lower() if job_description else ""
    
    # Step 1: Find which skills from the vocabulary appear in the JD
    jd_mentioned = []
    for skill in jd_vocabulary:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, jd_lower):
            jd_mentioned.append(skill)
    
    # Step 2: Which of those does Radhika actually have?
    matched = [s for s in jd_mentioned if s in candidate_skills]
    missing = [s for s in jd_mentioned if s not in candidate_skills]
    
    if not jd_mentioned:
        coverage_pct = 0
        verdict = "**Domain Mismatch:** No recognizable technology keywords found in this job description."
    else:
        coverage_pct = int((len(matched) / len(jd_mentioned)) * 100)
        if coverage_pct >= 70:
            verdict = "**Strong Match:** High skill coverage for this role."
        elif coverage_pct >= 40:
            verdict = "**Moderate Match:** Partial skill overlap with transferable foundations."
        else:
            verdict = "**Low Match:** Limited technology overlap with this job description."
    
    # Step 3: RAG-grounded qualitative assessment via FAISS + Gemini
    qualitative_section = ""
    try:
        vs = get_or_build_vectorstore()
        resume_chunks = vs.similarity_search(job_description, k=6)
        resume_context = "\n".join([doc.page_content for doc in resume_chunks])
        
        grounding_prompt = (
            "Based ONLY on the retrieved resume context below, write a 3-4 sentence "
            "qualitative fit assessment for this candidate against the job description. "
            "Highlight specific experience that maps to JD requirements, and note gaps honestly. "
            "Do NOT invent skills or experience not in the context.\n\n"
            f"Job Description:\n{job_description}\n\n"
            f"Retrieved Resume Context:\n{resume_context}\n\n"
            f"Matched Skills: {', '.join(matched) if matched else 'None'}\n"
            f"Missing Skills: {', '.join(missing) if missing else 'None'}\n\n"
            "Qualitative Assessment:"
        )

        api_key = get_default_api_key()
        model_candidates = ["gemini-flash-latest", "gemini-2.0-flash-lite", "gemini-3.6-flash", "gemini-2.0-flash", "gemini-pro"]
        llm = None
        for model_name in model_candidates:
            try:
                llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.2)
                qual_response = llm.invoke(grounding_prompt)
                qualitative_section = extract_clean_text(qual_response.content)
                break
            except Exception as ex:
                llm = None
                err_msg = str(ex)
                continue
                
        if not qualitative_section:
            qualitative_section = f"Qualitative assessment unavailable ({err_msg if 'err_msg' in locals() else 'API key required'})."
    except Exception as e:
        qualitative_section = f"Qualitative assessment unavailable ({str(e)})."
    
    return f"""
### Candidate Fit Report — Radhika Dhama

**Skill Coverage Score: {coverage_pct}%** ({len(matched)} / {len(jd_mentioned)} JD skills matched)

---

#### Skill Breakdown:
- **Matched:** {', '.join([f'`{s}`' for s in matched]) if matched else 'None'}
- **Missing:** {', '.join([f'`{s}`' for s in missing]) if missing else 'None — full coverage'}

#### RAG-Grounded Qualitative Assessment:
{qualitative_section}

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

def clean_markdown_output(text: str) -> str:
    cleaned = extract_clean_text(text)
    # Remove ASCII horizontal divider lines (===, ---, ___) that trigger Setext markdown header bugs
    cleaned = re.sub(r'^[=\-_]{3,}\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\n[=\-]{3,}\n', '\n\n', cleaned)
    return cleaned.strip()

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
        role = inputs.get("role", "")
        system_prompt = (
            "You are ResAgent, an executive AI copilot for Radhika Dhama's portfolio. "
            f"The recruiter is evaluating candidate fit for: {role}. "
            "Formulate responses in clean, beautifully structured Markdown with standard subheadings (###). "
            "Do NOT use ASCII divider lines such as '=======' or '-------'. "
            "Format links as clean hyperlinked Markdown text, e.g. [LinkedIn](https://...) and [GitHub](https://...). "
            "For ANY question about Radhika, call search_candidate_portfolio and answer accurately from retrieved context."
        )
        full_query = f"{system_prompt}\n\nUser Question: {user_text}"
        try:
            ai_msg = self.llm_with_tools.invoke(full_query)
            if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                tool_outputs = []
                tools_used = []
                for tool_call in ai_msg.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    if tool_name in self.tools:
                        tools_used.append(tool_name)
                        tool_func = self.tools[tool_name]
                        arg_val = list(tool_args.values())[0] if tool_args else user_text
                        tool_res = tool_func.invoke(arg_val)
                        tool_outputs.append(f"{tool_res}")
                context_str = "\n\n".join(tool_outputs)
                final_prompt = (
                    f"System: Respond in professional Markdown. Avoid ASCII lines like '======='. Format links as [Name](URL).\n"
                    f"Target Role: {role}\n"
                    f"User Question: {user_text}\n\n"
                    f"Retrieved Data:\n{context_str}\n\n"
                    f"Answer:"
                )
                final_ans = self.llm.invoke(final_prompt)
                return {"output": clean_markdown_output(final_ans.content), "tools_used": tools_used}
            else:
                return {"output": clean_markdown_output(ai_msg.content), "tools_used": []}
        except Exception as e:
            vs_docs = search_candidate_portfolio.invoke(user_text)
            clean_docs = clean_markdown_output(vs_docs)
            return {"output": f"### Candidate Summary — Radhika Dhama\n\n{clean_docs}", "tools_used": ["search_candidate_portfolio (fallback)"]}

def get_resagent_executor(api_key: str = None):
    return ResAgentEngine(api_key or get_default_api_key())
