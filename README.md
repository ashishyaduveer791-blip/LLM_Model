# 📚 AI Study Assistant using RAG

An AI-powered Study Assistant that allows students to learn directly from their own learning resources using Retrieval-Augmented Generation (RAG).

The system can process YouTube videos and PDF/books, retrieve the most relevant content for a student's question, and generate an easy-to-understand answer.

## 🚀 Features

- 🎥 Ask questions from YouTube videos
- 📄 Ask questions from PDF documents and books
- 🔎 Semantic search using vector embeddings
- 🧠 Retrieval-Augmented Generation (RAG)
- 📚 Page-aware PDF retrieval
- 💬 Multiple questions from the same document/video
- ⚡ FAISS vector database
- 🆓 Local embeddings using FastEmbed
- ☁️ Groq cloud LLM
- 🔄 Gemini fallback support
- 🔐 API keys stored securely using environment variables / Colab Secrets

## 🏗️ Architecture

YouTube / PDF
        ↓
Text Extraction
        ↓
Text Chunking
        ↓
FastEmbed Embeddings
        ↓
FAISS Vector Database
        ↓
User Question
        ↓
Similarity Search
        ↓
Relevant Context
        ↓
Groq / Gemini
        ↓
Final Answer

## 🛠️ Technologies Used

- Python
- LangChain
- FAISS
- FastEmbed
- Groq API
- Google Gemini API
- YouTube Transcript API
- PyPDF
- Google Colab

## 📖 Example

User uploads the book **Introduction to Algorithms** and asks:

> Teach me insertion sort from this book in simple language.

The system retrieves the relevant sections of the book and generates an explanation based on those sections.

## 🎯 Project Goal

The goal is to help students avoid manually searching through long lectures, books, PDFs, and notes.

Instead, students can ask for exactly what they want to learn and receive an explanation grounded in their learning material.

## 🔮 Future Improvements

- PPT/PPTX support
- Unified web interface
- Multiple document support
- Conversation history
- Source citations
- Better retrieval/reranking
- More LLM fallback providers
- User authentication
- Persistent vector database

## 🔐 Security

API keys are not stored directly in the source code.

For local development, use a `.env` file.

For Google Colab, use Colab Secrets.

Never commit API keys to GitHub.
