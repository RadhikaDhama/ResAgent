import streamlit as st
import os
import json
import urllib.request
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from agent_engine import get_resagent_executor, analyze_job_description_fit

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Page Configuration - Clean Executive Layout
st.set_page_config(
    page_title="Radhika Dhama — Portfolio",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Enhanced Radio Buttons, Fonts, Icons)
css_styles = """
<style>
    .stApp {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }
    div[data-testid="stRadio"] > label {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
        margin-bottom: 8px !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        margin-bottom: 6px !important;
        width: 100% !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
        border-color: #38BDF8 !important;
        background-color: #0F172A !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label p {
        font-size: 1.0rem !important;
        font-weight: 600 !important;
        color: #F8FAFC !important;
    }
    .header-container {
        padding: 0 0 16px 0;
        border-bottom: 1px solid #1E293B;
        margin-bottom: 20px;
    }
    .hero-title {
        font-size: 1.95rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: -0.4px;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #38BDF8;
        font-weight: 600;
        margin-top: 4px;
    }
    .sidebar-brand {
        padding: 4px 0 14px 0;
        border-bottom: 1px solid #1E293B;
        margin-bottom: 16px;
    }
    .sidebar-brand h2 {
        color: #F8FAFC;
        margin: 0;
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.3px;
    }
    .sidebar-brand p {
        color: #38BDF8;
        margin: 4px 0 0 0;
        font-size: 0.88rem;
        font-weight: 600;
    }
    .contact-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .contact-card-icon {
        width: 22px;
        height: 22px;
        flex-shrink: 0;
    }
    .contact-card-content {
        flex-grow: 1;
    }
    .contact-card-title {
        font-size: 0.70rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .contact-card-value {
        font-size: 0.88rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-top: 2px;
    }
    .contact-card-value a {
        color: #38BDF8;
        text-decoration: none;
    }
    .contact-card-value a:hover {
        text-decoration: underline;
    }
    .badge-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-left: 4px solid #38BDF8;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .badge-card-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-card-value {
        font-size: 1.10rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 2px;
    }
    .project-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 16px;
        height: 100%;
    }
    .project-card h3 {
        color: #F8FAFC;
        margin: 0 0 8px 0;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .tech-tag {
        display: inline-block;
        background: #0F172A;
        border: 1px solid #38BDF8;
        color: #38BDF8;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        margin: 0 4px 8px 0;
    }
    .copilot-banner {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .copilot-banner h3 {
        color: #F8FAFC;
        margin: 0 0 4px 0;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .copilot-banner p {
        color: #94A3B8;
        margin: 0;
        font-size: 0.92rem;
        line-height: 1.45;
    }
</style>
"""
st.markdown(css_styles, unsafe_allow_html=True)

# Helper Function: Live GitHub API Repository Fetcher
@st.cache_data(ttl=3600)
def fetch_github_repos(username="RadhikaDhama"):
    url = f"https://api.github.com/users/{username}/repos?sort=updated&per_page=30"
    req = urllib.request.Request(url, headers={"User-Agent": "ResAgent-Portfolio"})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                repos = []
                for item in data:
                    if not item.get("fork", False):
                        repos.append({
                            "name": item.get("name"),
                            "description": item.get("description") or "Project repository on Radhika Dhama's GitHub profile.",
                            "url": item.get("html_url"),
                            "language": item.get("language") or "Python",
                            "topics": item.get("topics", []),
                            "updated_at": item.get("updated_at", "")[:10]
                        })
                return repos
    except Exception:
        pass
    return []

def filter_repos_for_role(repos, role):
    if not repos:
        return []
    role_lower = role.lower()
    
    matched_repos = []
    for r in repos:
        text = f"{r['name']} {r['description']} {' '.join(r['topics'])} {r['language']}".lower()
        if "speech" in role_lower or "generative" in role_lower:
            if any(k in text for k in ["speech", "audio", "tts", "lora", "transformer", "pytorch", "llm", "whisper", "orpheus"]):
                matched_repos.append(r)
        elif "quantitative" in role_lower or "risk" in role_lower:
            if any(k in text for k in ["portfolio", "risk", "quant", "credit", "finance", "opt", "var", "slsqp", "statistics"]):
                matched_repos.append(r)
        elif "data scientist" in role_lower or "statistical" in role_lower:
            if any(k in text for k in ["data", "stat", "ml", "machine-learning", "xgboost", "gmm", "analysis", "r", "python"]):
                matched_repos.append(r)
        else: # HP / Full Stack / Default
            matched_repos.append(r)
            
    return matched_repos if matched_repos else repos

# Read Resume Text for Download Button
base_dir = os.path.dirname(os.path.abspath(__file__))
resume_txt_path = os.path.join(base_dir, "data", "resume_knowledge.txt")
if not os.path.exists(resume_txt_path):
    resume_txt_path = "resume_knowledge.txt"

resume_content = ""
if os.path.exists(resume_txt_path):
    with open(resume_txt_path, "r", encoding="utf-8") as f:
        resume_content = f.read()

# =========================================================
# SIDEBAR - NAVIGATION & CONTACT
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>Radhika Dhama</h2>
        <p>M.Sc. Data Science @ CMI</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Navigation Menu**")
    nav_selection = st.radio(
        "Navigation Menu:",
        [
            "ResAgent Copilot",
            "About Radhika",
            "Experience & Internships",
            "Academic & Risk Projects",
            "Technical Competencies",
            "JD Alignment Studio"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Recruiter Target Role**")
    target_role = st.selectbox(
        "Target Role:",
        [
            "Graduate Engineer - AI & Full Stack (HP)",
            "Speech & Generative AI ML Engineer",
            "Quantitative Analyst & Financial Risk",
            "Data Scientist & Statistical Modeler"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Contact & Profiles**")
    
    if resume_content:
        st.download_button(
            label="Download Resume (TXT)",
            data=resume_content,
            file_name="Radhika_Dhama_Resume.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.markdown("""
    <div class="contact-card">
        <svg class="contact-card-icon" fill="#38BDF8" viewBox="0 0 24 24"><path d="M20 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/></svg>
        <div class="contact-card-content">
            <div class="contact-card-title">Email</div>
            <div class="contact-card-value"><a href="mailto:radhika.dhama684@gmail.com">radhika.dhama684@gmail.com</a></div>
        </div>
    </div>
    <div class="contact-card">
        <svg class="contact-card-icon" fill="#38BDF8" viewBox="0 0 24 24"><path d="M6.62 10.79c1.44 2.83 3.76 5.14 6.59 6.59l2.2-2.2c.27-.27.67-.36 1.02-.24 1.12.37 2.33.57 3.57.57.55 0 1 .45 1 1V20c0 .55-.45 1-1 1-9.39 0-17-7.61-17-17 0-.55.45-1 1-1h3.5c.55 0 1 .45 1 1 0 1.25.2 2.45.57 3.57.11.35.03.74-.25 1.02l-2.2 2.2z"/></svg>
        <div class="contact-card-content">
            <div class="contact-card-title">Phone</div>
            <div class="contact-card-value"><a href="tel:+919720163553">+91 9720163553</a></div>
        </div>
    </div>
    <div class="contact-card">
        <svg class="contact-card-icon" fill="#0A66C2" viewBox="0 0 24 24"><path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.28 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.75M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z"/></svg>
        <div class="contact-card-content">
            <div class="contact-card-title">LinkedIn Profile</div>
            <div class="contact-card-value"><a href="https://www.linkedin.com/in/radhika-dhama/" target="_blank">linkedin.com/in/radhika-dhama</a></div>
        </div>
    </div>
    <div class="contact-card">
        <svg class="contact-card-icon" fill="#F8FAFC" viewBox="0 0 24 24"><path d="M12 2A10 10 0 0 0 2 12c0 4.42 2.87 8.17 6.84 9.5.5.08.66-.23.66-.5v-1.69c-2.77.6-3.36-1.34-3.36-1.34-.46-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.87 1.52 2.34 1.07 2.91.83.1-.65.35-1.09.63-1.34-2.22-.25-4.55-1.11-4.55-4.92 0-1.11.38-2 1.03-2.71-.1-.25-.45-1.29.1-2.64 0 0 .84-.27 2.75 1.02.79-.22 1.65-.33 2.5-.33.85 0 1.71.11 2.5.33 1.91-1.29 2.75-1.02 2.75-1.02.55 1.35.2 2.39.1 2.64.65.71 1.03 1.6 1.03 2.71 0 3.82-2.34 4.66-4.57 4.91.36.31.69.92.69 1.85V21c0 .27.16.59.67.5C19.14 20.16 22 16.42 22 12A10 10 0 0 0 12 2z"/></svg>
        <div class="contact-card-content">
            <div class="contact-card-title">GitHub Profile</div>
            <div class="contact-card-value"><a href="https://github.com/RadhikaDhama" target="_blank">github.com/RadhikaDhama</a></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# MAIN CONTENT SECTIONS
# =========================================================

# SECTION 1: RESAGENT COPILOT
if nav_selection == "ResAgent Copilot":
    st.markdown(f"""
    <div class="header-container">
        <div class="hero-title">Radhika Dhama — Portfolio Copilot</div>
        <div class="hero-subtitle">Perspective: <b>{target_role}</b> | LangChain Agent with FAISS Vector Search</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="copilot-banner">
        <h3>Welcome to ResAgent</h3>
        <p>Ask anything about Radhika's resume, academic research, Coriolis internship engineering, quantitative risk projects, or code implementations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_inq, col_clr = st.columns([8.2, 1.8])
    with col_inq:
        st.markdown("**Sample Inquiries for Selected Role:**")
    with col_clr:
        if st.button("Clear History", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "Welcome. I am **ResAgent**, Radhika's portfolio copilot. Ask anything about her resume, technical projects, or candidate fit."}]
            st.session_state.chat_history = []
            st.rerun()

    c1, c2, c3, c4 = st.columns(4)
    selected_prompt = None
    if "HP" in target_role or "Full Stack" in target_role:
        if c1.button("Evaluate Match for HP JD"):
            selected_prompt = "Evaluate Radhika's profile for the HP Graduate Engineer AI & Full Stack position."
        if c2.button("Hindi TTS LoRA Pipeline"):
            selected_prompt = "Tell me about her Hindi TTS fine-tuning project and 66.6% WER reduction."
        if c3.button("Statistical OOD Rejection"):
            selected_prompt = "Explain her GMM and Mahalanobis distance OOD rejection system."
        if c4.button("Streamlit Web Dashboard"):
            selected_prompt = "Show me her Customer Personality Streamlit Dashboard project details."
    elif "Speech" in target_role:
        if c1.button("Hindi TTS Orpheus-3B"):
            selected_prompt = "Explain her 3-stage LoRA fine-tuning for Hindi TTS and WER metrics."
        if c2.button("Speech Evaluation Benchmark"):
            selected_prompt = "How did she use Whisper ASR and ECAPA-TDNN for speech evaluation?"
        if c3.button("Show LoRA Code Snippet"):
            selected_prompt = "Show me the technical code snippet for her LoRA fine-tuning pipeline."
        if c4.button("SLM Physics QA Testing"):
            selected_prompt = "What solution-generation pipeline testing did she perform for SLM Physics Tutor?"
    elif "Quantitative" in target_role:
        if c1.button("Mean-Variance Portfolio"):
            selected_prompt = "Explain her Nifty portfolio optimization framework using SLSQP solvers."
        if c2.button("Value-at-Risk (VaR)"):
            selected_prompt = "How did she implement historical and parametric VaR/CVaR at 95% confidence?"
        if c3.button("CreditRisk Defaulters"):
            selected_prompt = "Explain her CreditRisk default prediction pipeline and SMOTE oversampling."
        if c4.button("Show Portfolio Code"):
            selected_prompt = "Show me the Python optimization code for her portfolio risk project."
    else:
        if c1.button("Academic Achievements"):
            selected_prompt = "What are Radhika's rank achievements in IIT JAM, GATE Stat, and CMI CGPA?"
        if c2.button("Accenture Strategy S4"):
            selected_prompt = "Tell me about her Accenture Strategy Connect Season 4 PPI award."
        if c3.button("GMM OOD Rejection Layer"):
            selected_prompt = "Explain her per-class Gaussian Mixture Models OOD rejection paper."
        if c4.button("Full Candidate Overview"):
            selected_prompt = "Provide a comprehensive summary of Radhika's background and core strengths."

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Welcome. I am **ResAgent**, Radhika's portfolio copilot. Ask anything about her resume, technical projects, or candidate fit."}]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.form(key="copilot_form", clear_on_submit=True):
        user_input = st.text_input("Ask anything about Radhika's resume, research, or portfolio:", placeholder="Ask about her skills, projects, education, or internship experience...")
        submit_button = st.form_submit_button("Ask ResAgent")

    prompt_to_run = user_input or selected_prompt
    if prompt_to_run:
        st.session_state.messages.append({"role": "user", "content": prompt_to_run})
        with st.chat_message("user"):
            st.markdown(prompt_to_run)
        with st.chat_message("assistant"):
            with st.spinner("ResAgent processing inquiry..."):
                try:
                    executor = get_resagent_executor()
                    response = executor.invoke({"input": prompt_to_run, "chat_history": st.session_state.chat_history})
                    answer = response["output"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.chat_history.append(HumanMessage(content=prompt_to_run))
                    st.session_state.chat_history.append(AIMessage(content=answer))
                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")


# SECTION 2: ABOUT RADHIKA
elif nav_selection == "About Radhika":
    st.markdown("""
    <div class="header-container">
        <div class="hero-title">About Radhika Dhama</div>
        <div class="hero-subtitle">M.Sc. Data Science Candidate @ Chennai Mathematical Institute (CMI)</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Radhika Dhama is a Master's student in Data Science at **Chennai Mathematical Institute (CMI)** with a strong statistical foundation from University of Delhi (Rank 1 Holder). She brings hands-on industry experience in Speech LLM Fine-Tuning (Orpheus-3B), Statistical OOD Detection (GMM + Mahalanobis Distance), Quantitative Portfolio Risk Optimization, and Full-Stack Agentic AI Development.
    """)
    
    st.subheader("Key Academic Ranks & Recognitions")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="badge-card">
            <div class="badge-card-title">IIT JAM Mathematical Statistics (2025)</div>
            <div class="badge-card-value">All India Rank (AIR) 66</div>
        </div>
        <div class="badge-card">
            <div class="badge-card-title">University of Delhi (2021-2024)</div>
            <div class="badge-card-value">College Rank 1 Holder (CGPA 9.33)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="badge-card">
            <div class="badge-card-title">GATE Statistics (2025)</div>
            <div class="badge-card-value">All India Rank (AIR) 142</div>
        </div>
        <div class="badge-card">
            <div class="badge-card-title">Accenture Strategy Connect S4 (2025)</div>
            <div class="badge-card-value">Finalist & PPI Recipient (ConsultIQ)</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.subheader("Education Background")
    st.markdown("""
    * **Chennai Mathematical Institute (CMI)** — *M.Sc. in Data Science (2025 – Present)* | **CGPA: 8.69 / 10.0**
      * *Coursework:* Machine Learning, Algorithm Design, Statistics with R/Python, Data Visualization, Linear Algebra.
    * **University of Delhi (Ramanujan College)** — *B.Sc. Statistics Hons (2021 – 2024)* | **CGPA: 9.33 / 10.0 (Rank 1)**
      * *Coursework:* Time Series Analysis, Statistical Inference, Operations Research, Linear Models, Stochastic Processes.
    """)


# SECTION 3: EXPERIENCE & INTERNSHIPS
elif nav_selection == "Experience & Internships":
    st.markdown("""
    <div class="header-container">
        <div class="hero-title">Industry Internship Experience</div>
        <div class="hero-subtitle">Coriolis Technologies Pvt Ltd — Data Science & ML Intern (Summer 2026)</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("1. Hindi TTS Fine-Tuning (Orpheus-3B) — PyTorch, LoRA, Unsloth", expanded=True):
        st.markdown("""
        * **Challenge:** Pre-trained English speech model suffered complete failure on Hindi text (92% Word Error Rate).
        * **Implementation:** Designed a **3-stage LoRA fine-tuning pipeline** (pronunciation alignment, speaker-voice consistency, and 7-emotion expressiveness).
        * **Impact:** Reduced Word Error Rate by **66.6%** (0.92 -> 0.31) and Character Error Rate (CER) by **82.4%** (0.85 -> 0.15).
        * **Evaluation:** Benchmarked on 500 samples using Whisper ASR, ECAPA-TDNN speaker embeddings, and wav2vec2 emotion classifier.
        """)
        
    with st.expander("2. Statistical OOD Detection for Document Classification — Scikit-learn, GMM", expanded=True):
        st.markdown("""
        * **Challenge:** Deep learning classifier produced high false-positive rates on out-of-distribution inputs across 41 classes.
        * **Implementation:** Built a statistically trained OOD rejection layer using **Per-class Gaussian Mixture Models (GMM) + Relative Mahalanobis Distance** on deep embeddings with a trained meta-classifier.
        * **Impact:** Achieved **99% OOD recall** and **99.59% overall accuracy**, increasing valid document classification accuracy from 75% to 89%.
        """)

    with st.expander("3. Data Quality Assurance for SLM Physics Tutor — LaTeX, JSON Schema", expanded=True):
        st.markdown("""
        * **Implementation:** Stress-tested the solution-generation pipeline for a Grade 8–12 Physics Tutor Small Language Model (SLM).
        * **Impact:** Surfaced LaTeX rendering bugs, JSON schema mismatches, and unit/dimensional inconsistencies, feeding structured bug reports back into training pipelines.
        """)


# SECTION 4: ACADEMIC & RISK PROJECTS (DYNAMIC GITHUB REPOS INTEGRATION)
elif nav_selection == "Academic & Risk Projects":
    st.markdown(f"""
    <div class="header-container">
        <div class="hero-title">Academic Projects & GitHub Repositories</div>
        <div class="hero-subtitle">Filtered Perspective: <b>{target_role}</b> | Live Integration with <a href="https://github.com/RadhikaDhama" target="_blank" style="color:#38BDF8;">github.com/RadhikaDhama</a></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Featured Core Projects")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown("""
        <div class="project-card">
            <h3>Customer Personality Analysis Dashboard</h3>
            <div>
                <span class="tech-tag">Python</span>
                <span class="tech-tag">Streamlit</span>
                <span class="tech-tag">Plotly</span>
                <span class="tech-tag">Pandas</span>
            </div>
            <p>Built an interactive web analytics dashboard examining 2,000+ customer demographics, spending behaviors, and campaign response metrics with dynamic KPI panels and correlation heatmaps.</p>
            <p><b>GitHub Repository:</b> <a href="https://github.com/RadhikaDhama/Customer-Personality-Analysis-Dashboard" target="_blank">RadhikaDhama/Customer-Personality-Analysis-Dashboard</a></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<div style="height:12px;"></div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="project-card">
            <h3>CreditRisk Defaulter Prediction</h3>
            <div>
                <span class="tech-tag">Python</span>
                <span class="tech-tag">XGBoost</span>
                <span class="tech-tag">LightGBM</span>
                <span class="tech-tag">SMOTE</span>
            </div>
            <p>Credit default prediction pipeline prioritizing Recall over raw accuracy to minimize missed defaulters. Handled severe class imbalance via SMOTE oversampling, achieving <b>97.4% accuracy</b>.</p>
        </div>
        """, unsafe_allow_html=True)
    with pcol2:
        st.markdown("""
        <div class="project-card">
            <h3>Portfolio Optimization & Risk Allocation</h3>
            <div>
                <span class="tech-tag">Python</span>
                <span class="tech-tag">SciPy (SLSQP)</span>
                <span class="tech-tag">EWMA</span>
                <span class="tech-tag">VaR / CVaR</span>
            </div>
            <p>Constrained mean-variance portfolio optimization framework on Nifty stocks comparing Max-Sharpe and Minimum-Variance portfolios via SLSQP solvers. Extended with historical & parametric VaR/CVaR at 95% confidence.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<div style="height:12px;"></div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="project-card">
            <h3>ResAgent Agentic AI Copilot</h3>
            <div>
                <span class="tech-tag">LangChain</span>
                <span class="tech-tag">Google Gemini</span>
                <span class="tech-tag">FAISS RAG</span>
                <span class="tech-tag">Streamlit</span>
            </div>
            <p>Interactive AI agent routing user questions between RAG vector search, heuristic job-match engines, and dynamic code architecture viewers with 100% fail-safe fallback logic.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🐙 Live Repositories from GitHub Profile")
    st.caption(f"Dynamically fetched live from github.com/RadhikaDhama and filtered for target role: '{target_role}'")
    
    with st.spinner("Fetching repositories live from GitHub API..."):
        all_repos = fetch_github_repos("RadhikaDhama")
        role_repos = filter_repos_for_role(all_repos, target_role)

    if role_repos:
        gcol1, gcol2 = st.columns(2)
        for idx, repo in enumerate(role_repos):
            col_target = gcol1 if idx % 2 == 0 else gcol2
            with col_target:
                st.markdown(f"""
                <div class="project-card">
                    <h3>{repo['name']}</h3>
                    <div>
                        <span class="tech-tag">{repo['language']}</span>
                        <span class="tech-tag">Updated: {repo['updated_at']}</span>
                    </div>
                    <p>{repo['description']}</p>
                    <p><b>Repository URL:</b> <a href="{repo['url']}" target="_blank">{repo['url']}</a></p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""<div style="height:12px;"></div>""", unsafe_allow_html=True)
    else:
        st.info("Visit Radhika's complete GitHub profile at [github.com/RadhikaDhama](https://github.com/RadhikaDhama) to explore all repositories.")


# SECTION 5: TECHNICAL COMPETENCIES
elif nav_selection == "Technical Competencies":
    st.markdown("""
    <div class="header-container">
        <div class="hero-title">Technical Competencies & Skill Matrix</div>
        <div class="hero-subtitle">Quantitative Statistics, Deep Learning, ML, & Full-Stack Engineering</div>
    </div>
    """, unsafe_allow_html=True)
    
    chart_type = st.radio("Select Visualization:", ["Intuitive Bar Chart", "Polar Radar Chart"], horizontal=True)
    
    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        if HAS_PLOTLY:
            categories = ['Statistical Inference', 'Deep Learning / LLMs', 'Machine Learning', 'Speech & Audio', 'Quant Risk', 'Full-Stack / Streamlit']
            scores = [95, 92, 90, 90, 88, 85]
            
            if chart_type == "Intuitive Bar Chart":
                df_chart = pd.DataFrame({'Competency': categories, 'Proficiency Score (%)': scores})
                fig = px.bar(
                    df_chart, 
                    x='Proficiency Score (%)', 
                    y='Competency', 
                    orientation='h',
                    text='Proficiency Score (%)'
                )
                fig.update_traces(
                    marker_color='#38BDF8',
                    texttemplate='%{text}%', 
                    textposition='inside', 
                    textfont=dict(size=14, color="#0F172A", family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial")
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F8FAFC"),
                    height=380,
                    margin=dict(l=10, r=20, t=10, b=10),
                    showlegend=False,
                    xaxis=dict(range=[0, 100], title="Proficiency Score (%)"),
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = go.Figure(data=go.Scatterpolar(r=scores, theta=categories, fill='toself', line_color='#38BDF8'))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100]),
                        bgcolor="#1E293B"
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F8FAFC"),
                    showlegend=False,
                    height=380
                )
                st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Primary Skill Distribution:**")
        st.markdown("- **Deep Learning & Speech:** PyTorch, LoRA, Unsloth, Transformers, Whisper (ASR), wav2vec2, ECAPA-TDNN")
        st.markdown("- **Machine Learning:** Scikit-learn, GMM, XGBoost, LightGBM, Random Forest, Clustering, SMOTE")
        st.markdown("- **LLMs & Agentic AI:** LangChain, LangGraph, RAG, Function Calling, FAISS, Gemini API")
        st.markdown("- **Statistics:** Hypothesis Testing, Linear Models, Statistical Inference, Time Series, Mahalanobis Distance")
        st.markdown("- **Programming & Web:** Python, R, SQL, Streamlit, Plotly, LaTeX, Git/GitHub, WandB")

    with st.expander("Interpretation Guide — How Skill Scores Are Calculated"):
        st.markdown("""
        * **95% — Statistical Inference & Theory:** Derived from University of Delhi Rank 1 (9.33 CGPA), IIT JAM AIR 66, and GATE Statistics AIR 142.
        * **92% — Deep Learning & LLMs:** Proven hands-on 3-stage LoRA fine-tuning of Orpheus-3B and LangChain RAG vector databases.
        * **90% — Machine Learning & OOD:** Demonstrated per-class GMM + Relative Mahalanobis Distance OOD detection achieving 99.59% accuracy.
        * **90% — Speech Processing:** Evaluated fine-tuned models on Whisper ASR, ECAPA-TDNN speaker embeddings, and wav2vec2 classifiers.
        * **88% — Quantitative Risk:** Constrained SLSQP portfolio optimization, EWMA covariance, and VaR/CVaR tail-risk modeling.
        * **85% — Full-Stack Streamlit:** Built production interactive web apps, dashboard pipelines, and API integrations.
        """)


# SECTION 6: JD ALIGNMENT STUDIO
elif nav_selection == "JD Alignment Studio":
    st.markdown("""
    <div class="header-container">
        <div class="hero-title">Job Description Alignment Studio</div>
        <div class="hero-subtitle">Evaluate candidate match score against target recruiter requirements</div>
    </div>
    """, unsafe_allow_html=True)
    
    jd_text = st.text_area("Paste target Job Description text here:", height=180, placeholder="Paste Job Description requirements, skills, and qualifications...")
    if st.button("Evaluate Fit"):
        if not jd_text.strip():
            st.warning("Please paste a Job Description before evaluating.")
        else:
            with st.spinner("Analyzing candidate alignment against JD..."):
                try:
                    report = analyze_job_description_fit.invoke(jd_text)
                    st.markdown(report)
                except Exception as e:
                    st.error(f"Error evaluating fit: {str(e)}")
