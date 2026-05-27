import os
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


BASE_DIR = Path(__file__).resolve().parents[1]
index_folder = BASE_DIR / "index" / "embeddings"
index_name = "breastfeeding_index"


api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set your OpenAI API key as an environment variable 'OPENAI_API_KEY'.")


st.set_page_config(
    page_title="Nursing Mothers Chatbot",
    page_icon="🍼",
    layout="centered",
    initial_sidebar_state="expanded",
)

accent_color = "#5bc0be"

with st.sidebar:
    st.markdown("## 💁‍♀️ About")
    st.write(
        "This friendly AI assistant helps answer breastfeeding, nutrition, and infant care questions with trusted expert guidance. "
        "All responses are informed by leading health organizations like the AAP and WHO."
    )
    st.write("---")
    st.markdown(
        "**Disclaimer:** This chatbot provides general information, not medical advice. Always consult your healthcare provider for personal concerns."
    )

st.markdown(
    f"<h1 style='text-align:center;color:{accent_color};font-size:2.5em'>🍼 Nursing Mothers Chatbot</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='text-align:center;color:#3aafa9;font-size:1.15em;'>Your expert, confidential support for all things nursing and infant care</div><br>",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models_and_db():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(
        str(index_folder),
        embeddings=embedding_model,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=api_key)
    return vector_db, llm


vector_db, llm = load_models_and_db()

if "history" not in st.session_state:
    st.session_state.history = []


def rag_prompt(context, question):
    return f"""
You are a breastfeeding expert. Using only the information in the provided sources, answer the user's question clearly, concisely, and kindly. Cite authoritative recommendations (AAP, WHO) if available. If sources disagree, explain briefly. If uncertain, say so.

Sources:
{context}

User Question: {question}

Helpful Answer:
""".strip()


with st.form("chat-form"):
    question = st.text_area(
        "Type your breastfeeding or infant care question below:",
        height=80,
        placeholder="e.g. How often should I breastfeed my newborn?",
    )
    submitted = st.form_submit_button("Ask AI", use_container_width=True)

if submitted and question:
    with st.spinner("Retrieving and generating expert answer..."):
        docs = vector_db.similarity_search(question, k=5)
        doc_contents = list({doc.page_content.strip() for doc in docs})
        context = "\n".join(doc_contents)
        prompt = rag_prompt(context, question)
        answer = llm.invoke(prompt).content.strip()
        st.session_state.history.append({"question": question, "answer": answer})


if st.session_state.history:
    st.markdown("---")
    st.markdown(f"<h3 style='color:{accent_color};'>Chat History</h3>", unsafe_allow_html=True)
    for i, entry in enumerate(reversed(st.session_state.history)):
        q_color = "#2c2c34" if i % 2 == 0 else "#5bc0be"
        a_color = accent_color if i % 2 == 0 else "#3aafa9"
        st.markdown(
            f"<div style='background-color:#f6f9fa;border-radius:10px;margin-bottom:6px;padding:10px'>"
            f"<b style='color:{q_color}'>You:</b><br>{entry['question']}"
            f"<hr><b style='color:{a_color}'>AI:</b><br>{entry['answer']}</div>",
            unsafe_allow_html=True,
        )

st.markdown(
    "<br><hr><div style='text-align:center;font-size:0.9em;color:#777;'>Nursing Mothers Chatbot &copy; 2025 | Powered by LangChain & GPT-4o</div>",
    unsafe_allow_html=True,
)