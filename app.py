import os, streamlit as st
from dotenv import load_dotenv
from extract_pdf import extract_text_from_pdf

# LangChain + vector store imports
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------- 1. Load your OpenAI key ----------
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY missing in .env")
    st.stop()

# ---------- 2. Read and split the ACFR PDF ----------
pdf_text = extract_text_from_pdf("TRS_ACFR_2024.pdf")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
docs = splitter.create_documents([pdf_text])

# ---------- 3. Build the in-memory vector store ----------
embeddings  = OpenAIEmbeddings()  # uses your key internally
vectorstore = DocArrayInMemorySearch.from_documents(docs, embeddings)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})

# ---------- 4. Set up the QA chain with strict prompt ----------
custom_prompt = PromptTemplate.from_template("""
You are a helpful assistant answering questions about the TRS ACFR 2024 report.
Use only the information provided in the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
""")

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o-mini"),
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# ---------- 5. Streamlit interface ----------
st.set_page_config(
    page_title="TRS Annual Comprehensive Financial Report",
    page_icon="📘"
)
st.title("📘 TRS Annual Comprehensive Financial Report")
st.markdown(
    "Ask me anything based on the 2024 TRS Annual Comprehensive Financial Report."
)

# initialise chat history if it doesn't exist
if "history" not in st.session_state:
    st.session_state.history = []

# show examples only when chat is empty
if not st.session_state.history:
    st.markdown(
        """
        **Try one of these example questions:**

        • *What was the pension fund’s annual return in 2024?*
        • *How much did employer contributions change from 2023 to 2024?*
        • *What discount rate was used for actuarial valuations?*
        • *What is the net position restricted for pensions?*
        • *How many active vs. retired members were reported?*
        """
    )

# --- chat input and response ---
user_q = st.chat_input("Type your question…")
if user_q:
    # Search with score filtering
    search_results = vectorstore.similarity_search_with_score(user_q, k=4)
    threshold = 0.65
    relevant_docs = [doc for doc, score in search_results if score >= threshold]

    if not relevant_docs:
        fallback_response = (
            "I'm here to help with questions about the TRS ACFR 2024 report. "
            "Try asking about fund balances, contributions, or investment performance."
        )
        st.session_state.history.append((user_q, fallback_response))

        # Optional: log unrelated questions
        with open("unanswered_questions.txt", "a") as f:
            f.write(user_q + "\n")
    else:
        result = qa_chain(user_q)
        st.session_state.history.append((user_q, result["result"]))

# display chat history
for user_msg, bot_msg in st.session_state.history:
    st.chat_message("user").markdown(user_msg)
    st.chat_message("assistant").markdown(bot_msg)
