from email import message
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load embedding model and FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = FAISS.load_local("faiss_index_bge_small", embedding_model, allow_dangerous_deserialization=True)

# Set your OpenAI API key (or use dotenv if you prefer)
load_dotenv(override = True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key = OPENAI_API_KEY)

def rag_qa(query, history, k=3):
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n---\n".join([doc.page_content for doc in docs])
    prompt = f"""
    Imagine You are a receptionist at our company, so you have to take everything into heart(strongly hold the information) to respond to the user questions about the comapany accurately.
    if you dont know you don't need to helucination instead reply that i am with a limited knowledge instead mail us, we will solve you query.

    Context:
    {context}

    Question: {query}
    Answer:
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}] + history + [{"role": "user", "content": query}]
    )
    answer = response.choices[0].message.content
    return answer

demo = gr.ChatInterface(
    fn=rag_qa,
    type = "messages",
    title="DataLegos Info Genie",
    description="Feel free to ask any queries related to our company :)"
)

demo.launch()

# To run the app, use the following command in your terminal:
# uvicorn app:app --host 127.0.0.1 --port 8000 --reload
# Then visit http://localhost:8000/docs in your browser. 