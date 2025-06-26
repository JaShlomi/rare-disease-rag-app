import os
import json
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="🧬 Rare Disease RAG Chatbot",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- DATA LOCATIONS ----
DATA_DIR = "data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "abstracts_faiss_index")
KG_PATH = os.path.join(DATA_DIR, "kg_definitional_data.json")
MIM2GENE_PATH = os.path.join(DATA_DIR, "mim2gene.txt")

# ---- SUPPORTED DISEASES ----
disease_name_to_ordo_uri = {
    "Cystic Fibrosis": "http://www.orpha.net/ORDO/Orphanet_586",
    "Huntington's Disease": "http://www.orpha.net/ORDO/Orphanet_418",
    "Duchenne Muscular Dystrophy": "http://www.orpha.net/ORDO/Orphanet_683",
    "Spinal Muscular Atrophy": "http://www.orpha.net/ORDO/Orphanet_84",
    "Hemophilia A": "http://www.orpha.net/ORDO/Orphanet_448",
    "Hemophilia B": "http://www.orpha.net/ORDO/Orphanet_447",
    "Gaucher Disease": "http://www.orpha.net/ORDO/Orphanet_355",
    "Pompe Disease": "http://www.orpha.net/ORDO/Orphanet_365",
    "Neurofibromatosis type 1": "http://www.orpha.net/ORDO/Orphanet_636",
    "Prader-Willi Syndrome": "http://www.orpha.net/ORDO/Orphanet_739",
    "Angelman Syndrome": "http://www.orpha.net/ORDO/Orphanet_526",
    "Rett Syndrome": "http://www.orpha.net/ORDO/Orphanet_802",
    "Fragile X Syndrome": "http://www.orpha.net/ORDO/Orphanet_908",
    "Phenylketonuria": "http://www.orpha.net/ORDO/Orphanet_716",
    "Alpha-1 Antitrypsin Deficiency": "http://www.orpha.net/ORDO/Orphanet_60",
    "Marfan Syndrome": "http://www.orpha.net/ORDO/Orphanet_284",
    "Ehlers-Danlos Syndrome, Hypermobile Type": "http://www.orpha.net/ORDO/Orphanet_98253",
    "Sickle Cell Anemia": "http://www.orpha.net/ORDO/Orphanet_232",
    "Thalassemia Major": "http://www.orpha.net/ORDO/Orphanet_821",
    "Crigler-Najjar Syndrome Type 1": "http://www.orpha.net/ORDO/Orphanet_792"
}
DISEASE_LIST = list(disease_name_to_ordo_uri.keys())
disease_list_display = ", ".join(DISEASE_LIST)

# ---- LOAD RESOURCES (CACHED) ----
@st.cache_resource(show_spinner="Loading data and vectorstore...")
def load_resources():
    # 1. Vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    # 2. KG
    with open(KG_PATH, "r", encoding="utf-8") as f:
        kg_definitional_data = json.load(f)
    # 3. mim2gene
    mim_gene_data = {}
    with open(MIM2GENE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                mim_number = parts[0]
                mim_entry_type = parts[1]
                entrez_gene_id = parts[2] if parts[2] else None
                approved_gene_symbol = parts[3] if parts[3] else None
                if approved_gene_symbol:
                    mim_gene_data[approved_gene_symbol.upper()] = {
                        "mim_number": mim_number,
                        "mim_entry_type": mim_entry_type,
                        "entrez_gene_id": entrez_gene_id,
                        "approved_gene_symbol": approved_gene_symbol
                    }
                if entrez_gene_id and entrez_gene_id not in mim_gene_data:
                    mim_gene_data[entrez_gene_id] = {
                        "mim_number": mim_number,
                        "mim_entry_type": mim_entry_type,
                        "entrez_gene_id": entrez_gene_id,
                        "approved_gene_symbol": approved_gene_symbol
                    }
    return vectorstore, kg_definitional_data, mim_gene_data

vectorstore, kg_definitional_data, mim_gene_data = load_resources()

# ---- HELPER FUNCTIONS ----
def extract_disease_from_query(query):
    query_lower = query.lower()
    for d in DISEASE_LIST:
        if d.lower() in query_lower:
            return d
    # Loose synonyms
    if "huntington" in query_lower:
        return "Huntington's Disease"
    if "sma" in query_lower or "spinal muscular atrophy" in query_lower:
        return "Spinal Muscular Atrophy"
    if "nf1" in query_lower:
        return "Neurofibromatosis type 1"
    if "aatd" in query_lower or "alpha-1 antitrypsin" in query_lower:
        return "Alpha-1 Antitrypsin Deficiency"
    if "fxs" in query_lower or "fragile x" in query_lower:
        return "Fragile X Syndrome"
    if "pku" in query_lower or "phenylketonuria" in query_lower:
        return "Phenylketonuria"
    return None

def extract_gene_or_mim_from_query(query):
    import re
    # Look for MIM number (6 digits) or gene symbol
    for match in re.findall(r'\b\d{6}\b', query):
        if match in mim_gene_data:
            return match
    for match in re.findall(r'\b[A-Z][A-Z0-9]{2,}\b', query, re.IGNORECASE):
        if match.upper() in mim_gene_data:
            return match.upper()
    return None

def get_blazegraph_context(query):
    disease = extract_disease_from_query(query)
    if not disease:
        return "No specific rare disease definitional data found in knowledge graph."
    ordo_uri = disease_name_to_ordo_uri.get(disease)
    kg_data = kg_definitional_data.get(ordo_uri)
    if kg_data:
        return (
            f"Disease Name: {kg_data.get('comment', kg_data.get('label', 'N/A'))}\n"
            f"URI: {kg_data.get('uri', 'N/A')}\n"
            f"Description: {kg_data.get('comment', 'N/A')}\n"
            f"DB Xref: {kg_data.get('dbXref', 'N/A')}\n"
            f"Parent Class: {kg_data.get('parentLabel', 'N/A')}"
        )
    return "No specific rare disease definitional data found in knowledge graph."

def get_mim_gene_context(query):
    identifier = extract_gene_or_mim_from_query(query)
    if identifier:
        gene_info = mim_gene_data.get(identifier)
        if gene_info:
            return (
                "MIM Number: " + gene_info.get('mim_number', 'N/A') + "\n" +
                "MIM Entry Type: " + gene_info.get('mim_entry_type', 'N/A') + "\n" +
                "Entrez Gene ID: " + gene_info.get('entrez_gene_id', 'N/A') + "\n" +
                "Approved Gene Symbol: " + gene_info.get('approved_gene_symbol', 'N/A')
            )
    return "No specific gene or MIM data found in mim2gene.txt."

# ---- LLM + PROMPT ----
# Remove "Evidence" from LLM prompt!
prompt_template = (
    "You are a helpful assistant specialized in rare diseases. "
    "You only answer questions about the following diseases: " + disease_list_display + ". "
    "If asked about anything else, reply: "
    "'Sorry, I can only answer questions about these 20 rare diseases: " + disease_list_display + ".'\n\n"
    "For supported questions:\n"
    "- Start with a 2–3 sentence summary (TL;DR).\n"
    "- Follow with a comprehensive, structured answer using clear headings, bullet points, and explanations.\n"
    "- Do not cite PubMed IDs (PMIDs) in your answer. The actual PMIDs will be shown separately in the Sources section.\n"
    "- If you do not know the answer from the provided context, say so and do not make up information.\n\n"
    "Context from Scientific Abstracts:\n"
    "{context}\n\n"
    "Structured Knowledge Graph Context (from Blazegraph):\n"
    "{blazegraph_context}\n\n"
    "Gene and MIM Context (from mim2gene.txt):\n"
    "{mim_gene_context}\n\n"
    "Question: {question}\n"
    "Answer:\n"
)
PROMPT = PromptTemplate.from_template(prompt_template)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    openai_api_key=os.environ.get("OPENAI_API_KEY", None)
)

# ---- SESSION STATE ----
if "history" not in st.session_state:
    st.session_state.history = []

# ---- SIDEBAR ----
with st.sidebar:
    st.header("Controls")
    if st.button("🗑️ Start New Chat"):
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.subheader("Diseases Covered")
    col1, col2 = st.columns(2)
    split_idx = len(DISEASE_LIST) // 2
    with col1:
        for d in DISEASE_LIST[:split_idx]:
            st.markdown(f"- {d}")
    with col2:
        for d in DISEASE_LIST[split_idx:]:
            st.markdown(f"- {d}")
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.85em;color:gray;'>Powered by FAISS, OpenAI GPT-4o (via LangChain), and open biomedical data sources: PubMed (NCBI), Orphanet (ORDO/HOOM), and mim2gene.</div>",
        unsafe_allow_html=True
    )

# ---- TOP UI (TITLE, BRIEF, DISCLAIMER) ----
st.title("🧬 Rare Disease RAG Chatbot")
st.caption("Ask a question about one of 20 rare diseases to get an evidence-backed answer (PubMed, ORDO, mim2gene).")
with st.expander("ℹ️ About this App", expanded=False):
    st.markdown("""
    This app uses **Retrieval-Augmented Generation (RAG)** for 20 rare diseases. Your questions are matched to ~20,000 PubMed abstracts using a FAISS vector database, plus structured knowledge from Orphanet and gene-disease mappings. All evidence is cited.
    """)
# More subtle disclaimer:
st.markdown(
    "<div style='font-size:0.92em; color: #666; border-left: 4px solid #e5c07b; padding: 0.7em 1em; background: #f7f5ee; margin-bottom: 1em;'>"
    "<strong>Disclaimer:</strong> This tool provides information for educational purposes only and is not medical advice. Please consult a medical professional for health-related questions."
    "</div>",
    unsafe_allow_html=True,
)

# ---- CHAT DISPLAY ----
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        # Show question above answer
        if m["role"] == "assistant" and m.get("question"):
            st.markdown(f"**Question:** {m['question']}")
        st.markdown(m["content"])
        if m.get("evidence"):
            with st.expander("📚 Sources"):
                for e in m["evidence"]:
                    st.markdown(e)

# ---- CHAT INPUT + MAIN QA LOGIC ----
user_query = st.chat_input("Ask a question about one of the 20 rare diseases...")

if user_query:
    st.session_state.history.append({"role": "user", "content": user_query})

    # Retrieve abstracts
    docs = vectorstore.similarity_search(user_query, k=10)
    context = "\n\n".join([d.page_content for d in docs])

    # Prepare evidence list
    evidence_list = []
    for d in docs:
        pmid = d.metadata.get("pmid", "")
        title = d.metadata.get("title", "")
        if pmid and title:
            evidence_list.append(f"- PMID: {pmid}, Title: {title[:100]}...")

    blazegraph_context = get_blazegraph_context(user_query)
    mim_gene_context = get_mim_gene_context(user_query)

    # Prepare the prompt and run LLM
    prompt_input = PROMPT.format(
        context=context,
        blazegraph_context=blazegraph_context,
        mim_gene_context=mim_gene_context,
        question=user_query
    )

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            try:
                answer = llm.invoke(prompt_input)
                answer_text = answer.content if hasattr(answer, "content") else str(answer)
            except Exception as e:
                answer_text = f"An error occurred: {e}"

            # Only show the user question once, above the answer
            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "evidence": evidence_list,
                    "question": user_query,
                }
            )
            st.markdown(f"**Question:** {user_query}")
            st.markdown(answer_text)
            if evidence_list:
                with st.expander("📚 Sources"):
                    for e in evidence_list:
                        st.markdown(e)
