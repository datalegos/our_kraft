from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-your-actual-api-key-here":
    print("❌ Please set your OPENAI_API_KEY in the .env file")
    exit(1)

# Load embedding model and FAISS index
print("🔄 Loading embedding model and FAISS index...")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = FAISS.load_local("vector_index", embedding_model, allow_dangerous_deserialization=True)
openai = OpenAI(api_key=OPENAI_API_KEY)

def test_rag_qa(query, k=5):
    """Test the RAG QA function"""
    # Get relevant context
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Enhanced system prompt
    system_prompt = f"""You are DataLegos' helpful AI assistant. Use the provided context to answer questions about the company accurately and conversationally.

Guidelines:
- Answer based on the context provided
- Be specific and helpful
- If you don't find relevant information in the context, say "I don't have that specific information. Please contact us at info@data-legos.com for more details."
- Keep responses concise but informative

Context:
{context}"""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
            max_tokens=300,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# Test queries
test_queries = [
    "What services does DataLegos offer?",
    "Who are the team members at DataLegos?",
    "What is the Career Catalyst program?",
    "How can I contact DataLegos?",
    "What industries does DataLegos work with?",
    "Tell me about Neo4j services",
    "What is Ravi Kiran Ponduri's background?"
]

print("🤖 Testing DataLegos Chatbot")
print("=" * 60)

for i, query in enumerate(test_queries, 1):
    print(f"\n{i}. Query: {query}")
    print("-" * 40)
    
    answer = test_rag_qa(query)
    print(f"Answer: {answer}")
    
    if i < len(test_queries):
        input("\nPress Enter to continue to next question...")

print("\n✅ Testing completed!")
print("Your chatbot should now provide much better answers!")
print("Run 'python app.py' to start the web interface.")