# My-First-RAG-Pipeline
# CampusBuzz: A Campus Event Guide RAG Pipeline 🎓

A **beginner-friendly implementation** of a Retrieval-Augmented Generation (RAG) pipeline that answers questions about campus events and activities from a PDF document.

## 📋 What is This?

This project demonstrates how to build a RAG system from scratch. RAG (Retrieval-Augmented Generation) is a technique where an AI model retrieves relevant information from a knowledge base and uses it to generate accurate, contextual answers.

In this case, the system:
1. Loads a **Campus Event Guide PDF** containing information about campus activities, events, deadlines, and organizations
2. Breaks it into searchable **chunks** 
3. Stores those chunks in a **vector database**
4. Retrieves relevant information based on user queries
5. Uses an LLM to generate natural, contextual answers

### Example Queries
The system can answer questions like:
- "When is the Spring Career Fair?"
- "What outdoor trips does the Rec Center offer?"
- "How do I start a new club?"
- "Are there any music-related events?"
- "When is the community service day?"

## 🏗️ How RAG Works

```
┌─────────────────┐
│  PDF Document   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ 1. Load & Split Text    │ (Document Loading)
└────────┬────────────────┘
         │
         ▼
┌──────────────────────────┐
│ 2. Convert to Embeddings │ (Vectorization)
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ 3. Store in Vector DB    │ (Chroma)
└────────┬─────────────────┘
         │
User Query → Retrieve Similar Chunks → LLM → Answer
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google API Key (for embeddings and LLM)
- Jupyter Notebook or Google Colab

### Installation

```bash
# Install required dependencies
pip install langchain-community pypdf langchain-text-splitters
pip install langchain-google-genai langchain-community
pip install chromadb
```

### Configuration

Set your Google API key:
```python
import os
from google.colab import userdata

os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
```

Or locally:
```python
import os
os.environ["GOOGLE_API_KEY"] = "your-api-key-here"
```

### Running the Pipeline

The notebook is organized into 6 sequential parts:

1. **Document Loading** - Load the PDF and verify it
2. **Text Splitting** - Break the document into manageable chunks
3. **Embeddings** - Convert text chunks into vector representations
4. **Vector Store** - Store embeddings in Chroma database
5. **Retrieval & RAG Chain** - Build the Q&A system
6. **Testing** - Query the system and evaluate results

Run each cell in order and complete any `TODO` sections.

## 📊 Project Structure

```
CampusBuzz_RAG_Pipeline/
├── README.md                                 # This file
├── CampusBuzz_RAG_Pipeline_1.ipynb          # Main notebook
├── Campus_Event_Guide_Spring_2026.pdf       # Source document

```

## 🔧 Technologies Used

| Technology | Purpose |
|-----------|---------|
| **LangChain** | Framework for building RAG applications |
| **Google Gemini** | Large Language Model for generating answers |
| **Google Embeddings** | Convert text to vector representations |
| **Chroma** | Vector database for storing & retrieving embeddings |
| **PyPDF** | PDF document loading |

## 📚 Key Learning Concepts

### 1. **Document Chunking**
Breaking long documents into smaller pieces (~300-500 tokens) helps the model stay within context limits and improves retrieval relevance.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
```

### 2. **Embeddings**
Embeddings convert text into numerical vectors in high-dimensional space. Similar texts have similar embeddings, enabling semantic search.

```python
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
```

### 3. **Vector Search**
Finding semantically similar chunks using cosine similarity or other distance metrics:

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retrieved_docs = retriever.invoke(user_query)
```

### 4. **RAG Chain**
Connecting retriever + prompt + LLM to generate context-aware answers:

```python
rag_chain = create_stuff_documents_chain(llm, prompt) | retriever
```

## 📖 Example Usage

```python
# Ask a question
query = "When is the Spring Career Fair and what should I bring?"
response = rag_chain.invoke(query)
print(response)
```

**Output:**
```
The Spring Career Fair is the largest recruiting event of the semester. 
It takes place on Wednesday, February 19 from 10am to 3pm in Morrison Auditorium. 
Business professional attire is required, and you should bring at least 
20 printed copies of your resume.
```

## 🧪 Testing & Validation

The notebook includes sections to:
- ✅ Verify document loading
- ✅ Inspect chunking quality
- ✅ Test retrieval results
- ✅ Run test queries
- ✅ Handle out-of-scope questions gracefully

**Pro tip:** Always inspect what chunks are actually retrieved! This habit helps debug why answers are inaccurate.

```python
retrieved_docs = retriever.invoke("What events are available?")
for doc in retrieved_docs:
    print(doc.page_content)
    print("---")
```

## 🎯 What I Learnt

✓ How to load and preprocess documents  
✓ How embeddings enable semantic search  
✓ How vector databases work  
✓ How to build RAG systems end-to-end  
✓ How to debug retrieval and generation issues  
✓ How LLMs use context to generate accurate answers  


