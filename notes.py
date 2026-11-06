from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS

from groq import Groq
from google import genai
from dotenv import load_dotenv

import os
import hashlib


# ==================================================
# 1. LOAD API KEYS
# ==================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

print("Groq API key loaded:", bool(groq_api_key))
print("Gemini API key loaded:", bool(google_api_key))

if not groq_api_key and not google_api_key:
    print("No cloud API key found.")
    exit()


# ==================================================
# 2. CREATE LLM CLIENTS
# ==================================================

groq_client = None
gemini_client = None

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)

if google_api_key:
    gemini_client = genai.Client(api_key=google_api_key)


# ==================================================
# 3. GET PDF PATH
# ==================================================

pdf_path = input(
    "\nEnter PDF file path: "
).strip()

# Remove PowerShell & if accidentally pasted
if pdf_path.startswith("&"):
    pdf_path = pdf_path[1:].strip()

# Remove quotes
pdf_path = pdf_path.strip('"').strip("'")


if not os.path.isfile(pdf_path):
    print("PDF file not found.")
    exit()


print("\nPDF found!")


# ==================================================
# 4. CREATE UNIQUE CACHE NAME
# ==================================================

absolute_path = os.path.abspath(pdf_path)

pdf_id = hashlib.md5(
    absolute_path.encode()
).hexdigest()

cache_root = "pdf_vector_cache"

os.makedirs(
    cache_root,
    exist_ok=True
)

cache_path = os.path.join(
    cache_root,
    pdf_id
)


# ==================================================
# 5. LOAD EMBEDDING MODEL
# ==================================================

print("Loading embedding model...")

embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# ==================================================
# 6. LOAD CACHE OR PROCESS PDF
# ==================================================

if os.path.exists(cache_path):

    print("\nCached PDF found!")
    print("Loading saved vector database...")

    vector_store = FAISS.load_local(
        cache_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("PDF cache loaded successfully!")


else:

    # ==============================================
    # READ PDF
    # ==============================================

    print("\nFirst time processing this PDF.")
    print("Reading PDF...")

    reader = PdfReader(pdf_path)

    print("Total pages:", len(reader.pages))

    documents = []


    # ==============================================
    # EXTRACT PAGE TEXT
    # ==============================================

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text and text.strip():

            documents.append(
                {
                    "text": text,
                    "page": page_number
                }
            )


    print(
        "Pages containing text:",
        len(documents)
    )


    # ==============================================
    # SPLIT EACH PAGE INTO CHUNKS
    # ==============================================

    print("\nCreating chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    chunks = []


    for document in documents:

        page_chunks = splitter.create_documents(
            [document["text"]],
            metadatas=[
                {
                    "page": document["page"],
                    "source": os.path.basename(pdf_path)
                }
            ]
        )

        chunks.extend(page_chunks)


    print("Total chunks:", len(chunks))


    # ==============================================
    # CREATE VECTOR DATABASE
    # ==============================================

    print("\nCreating embeddings + FAISS...")
    print("Large books can take some time on first run.")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )


    # ==============================================
    # SAVE CACHE
    # ==============================================

    vector_store.save_local(
        cache_path
    )

    print("PDF Vector Store created!")
    print("PDF cache saved successfully!")


# ==================================================
# 7. CREATE RETRIEVER
# ==================================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)


# ==================================================
# 8. READY
# ==================================================

print("\n======================================")
print("       PDF STUDY ASSISTANT READY")
print("======================================")

print("Ask questions from your PDF.")
print("Type 'exit' to stop.")


# ==================================================
# 9. QUESTION LOOP
# ==================================================

while True:

    question = input(
        "\nAsk a question from the PDF: "
    ).strip()


    if question.lower() == "exit":
        print("\nExiting PDF Study Assistant...")
        break


    if not question:
        print("Please enter a question.")
        continue


    # ==============================================
    # RETRIEVE RELEVANT CONTENT
    # ==============================================

    print("\nSearching the PDF...")

    relevant_docs = retriever.invoke(
        question
    )


    if not relevant_docs:
        print("No relevant content found.")
        continue


    # ==============================================
    # BUILD CONTEXT WITH PAGE NUMBERS
    # ==============================================

    context_parts = []

    pages_found = []


    for doc in relevant_docs:

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        pages_found.append(page)

        context_parts.append(
            f"[PDF Page {page}]\n"
            f"{doc.page_content}"
        )


    context = "\n\n".join(
        context_parts
    )


    # ==============================================
    # RAG PROMPT
    # ==============================================

    prompt = f"""
You are an AI Study Assistant.

The student uploaded a PDF/book.

Answer the student's question ONLY using the
retrieved PDF context below.

RULES:

1. Use only the PDF context.
2. Do not invent information.
3. Explain in simple language.
4. If asked to teach a topic, teach it step-by-step.
5. Preserve important definitions, algorithms,
   formulas, and examples from the PDF.
6. Mention relevant PDF page numbers when useful.
7. If the provided context does not contain the answer,
   say exactly:

"I could not find this information in the PDF."

PDF CONTEXT:

{context}

STUDENT QUESTION:

{question}

ANSWER:
"""


    answer_generated = False


    # ==================================================
    # 10. PRIMARY — GROQ
    # ==================================================

    if groq_client:

        print("Trying Groq...")

        try:

            response = (
                groq_client
                .chat
                .completions
                .create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.2
                )
            )


            answer = (
                response
                .choices[0]
                .message
                .content
            )


            if answer:

                print("\n--- FINAL ANSWER ---\n")
                print(answer)

                print(
                    "\nRetrieved PDF pages:",
                    sorted(set(pages_found))
                )

                print("\nAnswered by: Groq")

                answer_generated = True


        except Exception as e:

            print("\nGroq failed:")
            print(e)


    # ==================================================
    # 11. FALLBACK — GEMINI
    # ==================================================

    if (
        not answer_generated
        and gemini_client
    ):

        print("\nSwitching to Gemini...")

        try:

            response = (
                gemini_client
                .models
                .generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt
                )
            )


            if response.text:

                print("\n--- FINAL ANSWER ---\n")
                print(response.text)

                print(
                    "\nRetrieved PDF pages:",
                    sorted(set(pages_found))
                )

                print("\nAnswered by: Gemini")

                answer_generated = True


        except Exception as e:

            print("\nGemini failed:")
            print(e)


    # ==================================================
    # 12. BOTH FAILED
    # ==================================================

    if not answer_generated:

        print(
            "\nThe PDF retrieval worked, "
            "but the cloud LLM providers failed."
        )