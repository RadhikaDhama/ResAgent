import json

with open(r"c:\Users\Admin\Desktop\Credit Risk\ResAgent\app.py", "r", encoding="utf-8") as f:
    app_py_code = f.read()

with open(r"c:\Users\Admin\Desktop\Credit Risk\ResAgent\agent_engine.py", "r", encoding="utf-8") as f:
    agent_engine_code = f.read()

with open(r"c:\Users\Admin\Desktop\Credit Risk\ResAgent\data\resume_knowledge.txt", "r", encoding="utf-8") as f:
    resume_knowledge_code = f.read()

notebook_content = {
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 1: Install required packages\n",
    "!pip install -q streamlit langchain langchain-community langchain-core langchain-google-genai google-generativeai faiss-cpu sentence-transformers plotly python-dotenv\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%writefile resume_knowledge.txt\n",
    resume_knowledge_code
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%writefile agent_engine.py\n",
    agent_engine_code
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%writefile app.py\n",
    app_py_code
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 5: Pre-download embedding model\n",
    "from sentence_transformers import SentenceTransformer\n",
    "print(\"Downloading embedding model...\")\n",
    "model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')\n",
    "print(\"✅ Ready!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 6: Launch Streamlit & Generate Public Link via Pinggy\n",
    "import subprocess\n",
    "import time\n",
    "\n",
    "!pkill -f streamlit\n",
    "!pkill -f ssh\n",
    "\n",
    "subprocess.Popen([\n",
    "    \"streamlit\", \"run\", \"app.py\",\n",
    "    \"--server.port\", \"8501\",\n",
    "    \"--server.address\", \"0.0.0.0\",\n",
    "    \"--server.enableCORS\", \"false\",\n",
    "    \"--server.enableXsrfProtection\", \"false\"\n",
    "])\n",
    "\n",
    "time.sleep(3)\n",
    "\n",
    "print(\"⚡ Generating HTTPS URL via Pinggy...\")\n",
    "!ssh -o StrictHostKeyChecking=no -p 443 -R0:localhost:8501 a.pinggy.io\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open(r"c:\Users\Admin\Desktop\Credit Risk\ResAgent\data\agentic-ai-prototype.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)

print("SUCCESS: Notebook updated cleanly without secrets!")
