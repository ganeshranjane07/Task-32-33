import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load existing Chroma database
vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embeddings
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="mmr"
)

# Initialize LLM
llm = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq"
)

# System prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a TechNova Solutions company information assistant.

Answer ONLY using the provided knowledge base.
Give a complete, clear sentence as the answer.

If the answer is not available in the knowledge base,
say exactly: "I could not find the answer."

Do not use outside knowledge."""
    ),
    (
        "human",
        """Knowledge Base:
{context}

Question:
{question}"""
    )
])

print("TechNova RAG Chatbot")
print("Type 'exit' to quit.")

# Continuous chat
while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Chatbot stopped.")
        break

    # Retrieve relevant documents
    docs = retriever.invoke(question)

    # Create context
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # Create prompt
    messages = prompt.invoke({
        "context": context,
        "question": question
    })

    # Generate answer
    response = llm.invoke(messages)

    print("AI:", response.content)