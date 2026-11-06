from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

from groq import Groq
from google import genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

import os


# ==================================================
# 1. LOAD API KEYS
# ==================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

print("Groq API key loaded:", bool(groq_api_key))
print("Gemini API key loaded:", bool(google_api_key))

if not groq_api_key and not google_api_key:
    print("\nNo API key found.")
    print("Add GROQ_API_KEY or GOOGLE_API_KEY to .env")
    exit()


# ==================================================
# 2. CREATE CLOUD CLIENTS
# ==================================================

groq_client = None
gemini_client = None

if groq_api_key:
    try:
        groq_client = Groq(
            api_key=groq_api_key
        )
    except Exception as e:
        print("\nGroq client creation error:")
        print(e)


if google_api_key:
    try:
        gemini_client = genai.Client(
            api_key=google_api_key
        )
    except Exception as e:
        print("\nGemini client creation error:")
        print(e)


# ==================================================
# 3. GET YOUTUBE URL
# ==================================================

video_url = input(
    "\nEnter YouTube video link: "
).strip()

parsed_url = urlparse(video_url)


# ==================================================
# 4. EXTRACT VIDEO ID
# ==================================================

if "youtu.be" in parsed_url.netloc:

    video_id = (
        parsed_url.path
        .lstrip("/")
        .split("?")[0]
    )

elif "youtube.com" in parsed_url.netloc:

    video_id = parse_qs(
        parsed_url.query
    ).get("v", [None])[0]

else:

    video_id = None


if not video_id:
    print("\nInvalid YouTube URL.")
    exit()


print("Video ID:", video_id)


# ==================================================
# 5. LOAD LOCAL EMBEDDING MODEL
# ==================================================

print("\nLoading embedding model...")


try:

    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

except Exception as e:

    print("\nEmbedding model error:")
    print(e)
    exit()


# ==================================================
# 6. CREATE CACHE PATH
# ==================================================

cache_root = "vector_cache"

os.makedirs(
    cache_root,
    exist_ok=True
)

cache_path = os.path.join(
    cache_root,
    video_id
)


# ==================================================
# 7. LOAD CACHE OR PROCESS VIDEO
# ==================================================

if os.path.exists(cache_path):

    print("\nCached video found.")
    print("Loading saved vector database...")


    try:

        vector_store = FAISS.load_local(
            cache_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        print("Cache loaded successfully!")


    except Exception as e:

        print("\nCache loading error:")
        print(e)
        exit()


else:

    # ==============================================
    # FETCH TRANSCRIPT
    # ==============================================

    print("\nNo cache found.")
    print("Fetching YouTube transcript...")


    ytt_api = YouTubeTranscriptApi()


    try:

        transcript = ytt_api.fetch(
            video_id,
            languages=[
                "en-US",
                "en",
                "hi"
            ]
        )


    except Exception as e:

        print("\nTranscript Error:")
        print(e)
        exit()


    print("Transcript fetched successfully!")


    # ==============================================
    # TRANSCRIPT -> TEXT
    # ==============================================

    full_text = " ".join(
        snippet.text
        for snippet in transcript
    )


    if not full_text.strip():

        print("\nTranscript is empty.")
        exit()


    # ==============================================
    # TEXT -> CHUNKS
    # ==============================================

    print("\nCreating chunks...")


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )


    chunks = splitter.create_documents(
        [full_text]
    )


    print(
        "Total chunks:",
        len(chunks)
    )


    # ==============================================
    # CHUNKS -> FAISS
    # ==============================================

    print("\nCreating vector database...")


    try:

        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )


    except Exception as e:

        print("\nVector Store Error:")
        print(e)
        exit()


    print("Vector Store Created Successfully!")


    # ==============================================
    # SAVE FAISS
    # ==============================================

    try:

        vector_store.save_local(
            cache_path
        )

        print("Vector Store saved to cache!")


    except Exception as e:

        print("\nCache saving warning:")
        print(e)


# ==================================================
# 8. CREATE RETRIEVER
# ==================================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3
    }
)


# ==================================================
# 9. READY
# ==================================================

print("\n======================================")
print("       STUDY ASSISTANT READY")
print("======================================")

print("You can ask multiple questions.")
print("Type 'exit' to stop.")


# ==================================================
# 10. QUESTION LOOP
# ==================================================

while True:

    question = input(
        "\nAsk a question about the video: "
    ).strip()


    # ==============================================
    # EXIT
    # ==============================================

    if question.lower() == "exit":

        print("\nExiting Study Assistant...")
        break


    if not question:

        print("Please enter a question.")
        continue


    # ==============================================
    # RETRIEVE RELEVANT CHUNKS
    # ==============================================

    print("\nSearching video...")


    try:

        relevant_docs = retriever.invoke(
            question
        )


    except Exception as e:

        print("\nRetrieval Error:")
        print(e)
        continue


    if not relevant_docs:

        print(
            "\nNo relevant information found "
            "in the video."
        )

        continue


    # ==============================================
    # CREATE CONTEXT
    # ==============================================

    context = "\n\n".join(
        doc.page_content
        for doc in relevant_docs
    )


    # ==============================================
    # CREATE RAG PROMPT
    # ==============================================

    prompt = f"""
You are a helpful AI Study Assistant.

Answer the user's question using ONLY the retrieved
YouTube video transcript context.

RULES:

1. Use only the VIDEO CONTEXT.
2. Do not invent information.
3. Explain clearly and simply.
4. If the user asks you to teach a topic,
   explain it step-by-step.
5. Use examples from the context when useful.
6. If the answer is not available in the context,
   say exactly:

"I could not find this information in the video."


VIDEO CONTEXT:

{context}


USER QUESTION:

{question}


ANSWER:
"""


    answer_generated = False


    # ==================================================
    # 11. PRIMARY PROVIDER: GROQ
    # ==================================================

    if groq_client:

        print("\nTrying Groq...")


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
                    "\nAnswered by: Groq"
                )

                answer_generated = True


        except Exception as e:

            print("\n==============================")
            print("GROQ ERROR")
            print("==============================")

            print(e)

            print(
                "\nGroq failed. "
                "Switching to Gemini..."
            )


    # ==================================================
    # 12. FALLBACK PROVIDER: GEMINI
    # ==================================================

    if (
        not answer_generated
        and gemini_client
    ):

        print("\nTrying Gemini fallback...")


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
                    "\nAnswered by: Gemini"
                )

                answer_generated = True


        except Exception as e:

            print("\n==============================")
            print("GEMINI ERROR")
            print("==============================")

            print(e)


    # ==================================================
    # 13. BOTH FAILED
    # ==================================================

    if not answer_generated:

        print("\n==============================")
        print("ALL PROVIDERS FAILED")
        print("==============================")

        print(
            "\nThe RAG retrieval worked, "
            "but the cloud LLMs could not answer."
        )

        print(
            "You can ask another question "
            "without processing the video again."
        )