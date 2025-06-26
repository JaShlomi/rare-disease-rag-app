# 🧬 Rare Disease RAG Chatbot

**Live demo:** [https://rare-disease-rag-app-etaz9a8nphysbuuhxlyj85.streamlit.app/](https://rare-disease-rag-app-etaz9a8nphysbuuhxlyj85.streamlit.app/)

## 🚀 Overview

This project is a Retrieval-Augmented Generation (RAG) Streamlit chatbot focused on 20 rare diseases. It brings together:

- **PubMed abstracts** (~18,000, 2020–2024)
- **Orphanet/ORDO/HOOM knowledge graph data**
- **Gene–disease mappings (mim2gene)**
- **FAISS vector search with BioBERT embeddings**
- **OpenAI GPT-4o for answer generation**
- **Works locally and on Streamlit Cloud**

The chatbot provides evidence-backed, referenced answers to questions about rare diseases using recent scientific literature and curated structured knowledge.

---

## 🧩 Key Components

- **Knowledge Graph:** Data loaded from ORDO/HOOM OWL files into a local Blazegraph instance, with Python scripts to extract key facts (no live SPARQL in the app).
- **Literature RAG:** PubMed abstracts downloaded, cleaned, and embedded with BioBERT. FAISS is used for semantic retrieval.
- **Chatbot UI:** Streamlit app with modern chat interface.
- **Gene Mapping:** Uses `mim2gene.txt` for gene–disease relationships.

---

## ⚡️ Quick Demo

- [Live app on Streamlit Cloud](https://rare-disease-rag-app-etaz9a8nphysbuuhxlyj85.streamlit.app/)

**Try these sample questions:**

1. What are the main symptoms of Huntington's Disease?
2. How is Gaucher Disease diagnosed?
3. Which gene is mutated in Cystic Fibrosis?
4. What are the latest advances in exon-skipping therapies for Duchenne Muscular Dystrophy?
5. How is newborn screening for Cystic Fibrosis performed?

---

## 🔄 Workflow Summary

**How the system was built:**

1. **Orphanet/ORDO Data Extraction**
   - Loaded OWL files into Blazegraph locally.
   - Wrote Python scripts to extract definitional disease data as JSON for each disease.
   - **Note:** The app uses pre-extracted data, not live SPARQL.

2. **PubMed Literature Fetching**
   - Downloaded up to 1,000 abstracts per disease (not limited to reviews).
   - Cleaned, deduplicated, and stored as JSON.

3. **Embeddings & FAISS Indexing**
   - Used BioBERT (`pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`) via HuggingFace + LangChain.
   - Indexed all abstracts in a FAISS vectorstore (`/data/abstracts_faiss_index`).

4. **Streamlit RAG App**
   - Loads FAISS index, disease JSON, and mim2gene.
   - User’s question triggers a vector search (top k=10), showing PMIDs as sources.
   - LLM generates a structured, referenced answer using only retrieved context.
   - **App** appends the evidence list, not the LLM (to prevent “hallucinated” PMIDs).

5. **Deployment**
   - Works locally (Python 3.12.x recommended).
   - Deployed to [Streamlit Cloud](https://rare-disease-rag-app-etaz9a8nphysbuuhxlyj85.streamlit.app/).

---

## 🛠️ Local Setup

1. **Clone the repo:**

    ```bash
    git clone https://github.com/<your-username>/<your-repo>.git
    cd <your-repo>
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Add your OpenAI API key**  
   - Use an `.env` file, `secrets.toml`, or export as an environment variable:
     ```
     export OPENAI_API_KEY=sk-...
     ```

4. **Run the app:**

    ```bash
    streamlit run streamlit_app.py
    ```

5. **Data directory:**  
   - `data/abstracts_faiss_index` (FAISS vectorstore, must be present)
   - `data/kg_definitional_data.json` (pre-extracted KG)
   - `data/mim2gene.txt`

---

## 🌐 Streamlit Cloud Deployment

- Push code and data files to your GitHub repo.
- On [Streamlit Cloud](https://streamlit.io/cloud), connect your repo and set `OPENAI_API_KEY` in the secrets.
- Python 3.12 is recommended.
- All `/data` files must be committed.

---

## 📄 Example Questions to Try

- What are the main symptoms of Huntington's Disease?
- How is Gaucher Disease diagnosed?
- Which gene is mutated in Cystic Fibrosis?
- What are the latest advances in exon-skipping therapies for Duchenne Muscular Dystrophy?
- How is newborn screening for Cystic Fibrosis performed?
- What are the most common GBA gene mutations in Gaucher Disease?
- What gene therapies are available for Spinal Muscular Atrophy?
- What treatment options exist for Fragile X Syndrome?
- What are typical complications of Marfan Syndrome?
- What are the clinical features of Rett Syndrome?

---

## 📝 Notes & Differences from the "Plan"

- No live SPARQL: Instead, KG data is pre-extracted as JSON.
- PubMed abstracts: All recent article types included (not just reviews).
- BioBERT used for embedding and retrieval.
- Sources appended only by the app, to ensure citation integrity.

---

## 📢 Credits

- Project by Shlomi Serge Jakubowicz, 2025
- Biomedical data: NCBI PubMed, Orphanet (ORDO), HOOM, mim2gene
- Vector search: FAISS + HuggingFace BioBERT + LangChain
- Chat: OpenAI GPT-4o via LangChain, Streamlit

---

## 🗣️ Questions, Feedback, Contributions?

Open an issue or PR in this repo, or [contact the author](mailto:jshlomi81@email.com).
