import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# ── STEP 1: Load Models ──
print("Loading models...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── STEP 2: Read PDF and extract text ──
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# ── STEP 3: Split text into chunks ──
def split_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# ── STEP 4: Set up Vector Database ──
chroma_client = chromadb.PersistentClient(
    path="D:/AI Engineer/chatpdf_db"
)
collection = chroma_client.get_or_create_collection("pdf_docs")

# ── STEP 5: Load PDFs into Vector DB ──
if collection.count() == 0:
    print("Reading PDFs...")
    
    pdf_files = [
        "D:/AI Engineer/unit1.pdf",
        "D:/AI Engineer/unit2.pdf",
        "D:/AI Engineer/rag_architectures.pdf"
    ]
    
    all_chunks = []
    all_ids = []
    chunk_id = 0
    
    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        chunks = split_into_chunks(text)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_id}")
            chunk_id += 1
    
    print(f"Total chunks created: {len(all_chunks)}")
    
    # Convert to embeddings and store
    embeddings = embedding_model.encode(all_chunks).tolist()
    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids
    )
    print(f"✅ Stored {len(all_chunks)} chunks in Vector DB!")
else:
    print(f"✅ Using existing {collection.count()} chunks!")

# ── STEP 6: RAG Function ──
def ask_pdf(question):
    # Search Vector DB
    question_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )
    relevant_chunks = results['documents'][0]
    
    # Build prompt with context
    context = "\n\n".join(relevant_chunks)
    prompt = f"""You are a helpful study assistant.
Answer the question based ONLY on the provided context.
If the answer is not in the context, say "This topic is not covered in the PDF."

Context from PDF:
{context}

Question: {question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content, relevant_chunks

# ── STEP 7: Chat Loop ──
print("\n📄 ChatPDF Ready! Ask questions about your ML notes!")
print("Type 'quit' to exit")
print("=" * 50)

while True:
    question = input("\nYou: ")
    if question.lower() == "quit":
        break
    
    answer, chunks = ask_pdf(question)
    print(f"\n🤖 Answer: {answer}")
    print("-" * 50)