**What does it do:**
1. You put some files into RAG/data. It can be PDF or TXT.
2. RAG will convert them into chunks locally.
3. RAG will send those chunks to OpenAI. You get back chunk vectors (as vector store index) which is stored in chroma_db/ locally.
4. When you ask question. RAG send that question to OpenAI which gives back a question vector.
5. RAG will compare the question vector with all chunk vectors to find the most similar chunk(s).
6. RAG will send those chosen chunks to OpenAI. You get back augmented answer.

----------------------------------

**How to Run Locally:**

**Download Python dependencies:**
1. python3 -m venv env
2. source env/bin/activate
3. pip3 install -r requirements.txt

**Build the index:**

python src/rag_os_agent.py --index

**Once built, run the agent:**

python src/rag_os_agent.py
