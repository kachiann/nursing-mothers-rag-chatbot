import os
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from tools import lookup_nutrient_info, check_breastfeeding_guideline

BASE_DIR = Path(__file__).resolve().parents[1]
index_folder = BASE_DIR / "index" / "embeddings"
index_name = "breastfeeding_index"

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set your OpenAI API key as an environment variable 'OPENAI_API_KEY'.")

TOOLS = [lookup_nutrient_info, check_breastfeeding_guideline]

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a knowledgeable, compassionate assistant for nursing mothers \
and caregivers of infants. You have access to two tools:

• lookup_nutrient_info — use when the user asks about the nutritional content \
of a specific food (e.g. calcium in spinach, protein in lentils).
• check_breastfeeding_guideline — use when the user asks about safety, \
recommendations, storage, duration, or best practices.

For all other questions, answer using the retrieved context below from \
authoritative medical sources (AAP, WHO, NHS, CDC).

Always clarify that your answers are informational and not a substitute \
for professional medical advice.

Retrieved context:
{context}""",
    ),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])


# -- PAGE CONFIG --
st.set_page_config(
    page_title="Nursing Mothers Chatbot",
    page_icon="🍼",
    layout="centered",
    initial_sidebar_state="expanded",
)

accent_color = "#5bc0be"

# -- SIDEBAR --
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
    st.write("---")
    st.markdown("**Tools available:**")
    st.markdown("🔬 Live USDA nutrient lookup")
    st.markdown("📋 WHO/AAP/CDC guideline lookup")

# -- HEADER --
st.markdown(
    f"<h1 style='text-align:center;color:{accent_color};font-size:2.5em'>🍼 Nursing Mothers Chatbot</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='text-align:center;color:#3aafa9;font-size:1.15em;'>Your expert, confidential support for all things nursing and infant care</div><br>",
    unsafe_allow_html=True,
)


# -- LOAD MODELS AND BUILD AGENT --
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
    agent = create_openai_tools_agent(llm, TOOLS, AGENT_PROMPT)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        max_iterations=4,
        handle_parsing_errors=True,
    )
    return vector_db, agent_executor


vector_db, agent_executor = load_models_and_db()

# -- SESSION STATE --
if "history" not in st.session_state:
    st.session_state.history = []
if "lc_chat_history" not in st.session_state:
    st.session_state.lc_chat_history = []


def get_context(question: str, k: int = 5) -> str:
    docs = vector_db.similarity_search(question, k=k)
    seen = set()
    unique = []
    for doc in docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique.append(content)
    return "\n\n".join(unique)


# -- CHAT FORM --
with st.form("chat-form"):
    question = st.text_area(
        "Type your breastfeeding or infant care question below:",
        height=80,
        placeholder="e.g. How often should I breastfeed my newborn?",
    )
    submitted = st.form_submit_button("Ask AI", use_container_width=True)

if submitted and question:
    with st.spinner("Retrieving and generating expert answer..."):
        context = get_context(question)
        result = agent_executor.invoke({
            "input": question,
            "context": context,
            "chat_history": st.session_state.lc_chat_history,
        })
        answer = result["output"]
        st.session_state.history.append({"question": question, "answer": answer})
        st.session_state.lc_chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer),
        ])

# -- CHAT HISTORY --
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

# -- FOOTER --
st.markdown(
    "<br><hr><div style='text-align:center;font-size:0.9em;color:#777;'>Nursing Mothers Chatbot &copy; 2025 | Powered by LangChain, GPT-4o & USDA FoodData Central</div>",
    unsafe_allow_html=True,
)
