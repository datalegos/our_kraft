from email import message
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
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

def rag_qa(query, history, k=5):
    # Get relevant context with higher k for better coverage
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # More concise system prompt to save tokens
    system_prompt = f"""You are DataLegos' AI assistant. Answer questions about the company using the provided context. If information isn't available, say "I don't have that information. Please contact us at info@data-legos.com"

Context:
{context}"""
    
    # Only use recent history (last 4 messages) to save tokens
    recent_history = history[-4:] if len(history) > 4 else history
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}] + recent_history + [{"role": "user", "content": query}],
        max_tokens=300,  # Limit response length
        temperature=0.3  # More focused responses
    )
    answer = response.choices[0].message.content
    return answer

demo = gr.ChatInterface(
    fn=rag_qa,
    type = "messages",
    title="DataLegos Info Genie",
    description="Feel free to ask any queries related to our company :)"
)

demo.launch(server_name="127.0.0.1", server_port=7860)

# Your Gradio app will run at http://127.0.0.1:7860 