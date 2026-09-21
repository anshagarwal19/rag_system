import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()



DB_DIR = "chroma-db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash-lite"


def configure_api_key():
    if os.getenv("GOOGLE_API_KEY"):
        return

    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except (FileNotFoundError, KeyError):
        pass

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)



st.markdown(
    """
    <style>

    /* =====================================================
       BASE APP
       ===================================================== */

    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background-color: #0b0d12 !important;
        color: #f4f6fb !important;
    }

    [data-testid="stHeader"] {
        background-color: #0b0d12 !important;
    }

    #MainMenu,
    footer,
    header {
        visibility: hidden;
    }

    .main .block-container {
        max-width: 1180px;
        padding: 2.5rem 3rem 7rem;
    }


    /* =====================================================
       GLOBAL TEXT
       ===================================================== */

    .stApp p,
    .stApp span,
    .stApp label,
    .stApp li,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {
        color: #f4f6fb !important;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    [data-testid="stSidebar"] {
        background-color: #10131a !important;
        border-right: 1px solid #252a35 !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: #10131a !important;
        padding: 1.5rem 1.15rem;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #e8ecf4 !important;
    }

    .brand {
        color: #ffffff !important;
        font-size: 1.2rem;
        font-weight: 750;
        letter-spacing: -0.025em;
    }

    .brand-subtitle {
        color: #858e9f !important;
        font-size: 0.82rem;
        margin-top: 4px;
        margin-bottom: 24px;
    }

    .section-title {
        color: #929bad !important;
        font-size: 0.73rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        margin-top: 24px;
        margin-bottom: 10px;
    }


    /* =====================================================
       SIDEBAR DIVIDERS
       ===================================================== */

    [data-testid="stSidebar"] hr {
        border-color: #252a35 !important;
    }


    /* =====================================================
       PDF UPLOADER
       ===================================================== */

    [data-testid="stFileUploader"] {
        background: #151922 !important;
        border: 1px dashed #3b4251 !important;
        border-radius: 14px !important;
        padding: 8px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #151922 !important;
        border: none !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #b8c0ce !important;
    }

    [data-testid="stFileUploader"] button {
        background: #252b36 !important;
        color: #f4f6fb !important;
        border: 1px solid #3a4250 !important;
    }

    [data-testid="stFileUploaderFileName"] {
        color: #f4f6fb !important;
    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {
        width: 100%;
        min-height: 42px;
        border-radius: 10px !important;

        background: #191d26 !important;
        color: #f4f6fb !important;

        border: 1px solid #343b49 !important;

        font-weight: 600;
    }

    .stButton > button:hover {
        background: #242936 !important;
        color: #ffffff !important;
        border-color: #596275 !important;
    }

    /* Primary button */

    .stButton > button[kind="primary"] {
        background: #f4f6fb !important;
        color: #10131a !important;
        border-color: #f4f6fb !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #ffffff !important;
        color: #000000 !important;
    }


    /* =====================================================
       HERO
       ===================================================== */

    .hero {
        padding-top: 0.3rem;
        padding-bottom: 2rem;
    }

    .hero h1 {
        color: #ffffff !important;

        font-size: 2.65rem;
        line-height: 1.05;

        margin: 0;

        letter-spacing: -0.05em;
    }

    .hero p {
        color: #8e97a8 !important;

        margin-top: 0.7rem;

        font-size: 1rem;
    }


    /* =====================================================
       CHAT MESSAGES
       ===================================================== */

    [data-testid="stChatMessage"] {
        background: #141820 !important;

        border: 1px solid #292f3b !important;

        border-radius: 16px !important;

        padding: 1rem 1.1rem !important;

        margin: 0.8rem 0 !important;

        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.18) !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        color: #e9edf5 !important;

        line-height: 1.65 !important;
    }


    /* User message */

    [data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-user"]
    ) {
        background: #1b202a !important;

        border-color: #303744 !important;
    }


    /* =====================================================
       CHAT INPUT
       ===================================================== */

    [data-testid="stChatInput"] {
        background: #0b0d12 !important;

        border-top: 1px solid #252a35 !important;

        padding-top: 12px !important;
    }

    [data-testid="stChatInput"] > div {
        background: #181c25 !important;

        border: 1px solid #363d4b !important;

        border-radius: 14px !important;

        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.3) !important;
    }

    [data-testid="stChatInput"] textarea {
        background: transparent !important;

        color: #f4f6fb !important;

        caret-color: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #747d8f !important;

        opacity: 1 !important;
    }

    [data-testid="stChatInput"] button {
        background: #292f3b !important;

        color: #ffffff !important;

        border-radius: 10px !important;
    }


    /* =====================================================
       SOURCES / EXPANDER
       ===================================================== */

    [data-testid="stExpander"] {
        background: #10131a !important;

        border: 1px solid #292f3b !important;

        border-radius: 11px !important;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] p {
        color: #b7c0cf !important;
    }

    .source-pill {
        display: inline-block;

        background: #1b202a;

        color: #aeb7c6 !important;

        border: 1px solid #303744;

        border-radius: 999px;

        padding: 4px 9px;

        margin: 3px 4px 0 0;

        font-size: 0.76rem;
    }


    /* =====================================================
       CAPTIONS
       ===================================================== */

    [data-testid="stCaptionContainer"],
    .stCaption {
        color: #858e9f !important;
    }


    /* =====================================================
       PROGRESS BAR
       ===================================================== */

    [data-testid="stProgressBar"] {
        background: #202530 !important;
    }


    /* =====================================================
       ALERTS
       ===================================================== */

    [data-testid="stAlert"] {
        border-radius: 10px !important;
    }


    /* =====================================================
       SCROLLBAR
       ===================================================== */

    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0b0d12;
    }

    ::-webkit-scrollbar-thumb {
        background: #303644;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #454d5d;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# RAG COMPONENTS
# =========================================================

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


@st.cache_resource
def get_vector_store():
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )


@st.cache_resource
def get_llm():
    configure_api_key()

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL
    )


def get_retriever():

    return get_vector_store().as_retriever(
        search_type="mmr",

        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )


# =========================================================
# PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use only the provided context for answering the question.

If the answer is not present in the context,
say:

"I could not find the answer in the document."
""",
        ),

        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)


# =========================================================
# PDF INGESTION
# =========================================================

def add_pdf_to_database(uploaded_file):

    suffix = Path(
        uploaded_file.name
    ).suffix or ".pdf"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as tmp:

        tmp.write(
            uploaded_file.getvalue()
        )

        temp_path = tmp.name

    try:

        docs = PyPDFLoader(
            temp_path
        ).load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        chunks = splitter.split_documents(
            docs
        )

        if not chunks:
            return 0

        vector_store = get_vector_store()

        vector_store.add_documents(
            chunks
        )

        return len(chunks)

    finally:

        try:
            os.remove(temp_path)

        except OSError:
            pass


# =========================================================
# QUESTION ANSWERING
# =========================================================

def answer_question(query):

    retriever = get_retriever()

    docs = retriever.invoke(
        query
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query,
        }
    )

    response = get_llm().invoke(
        final_prompt
    )

    return response.content, docs


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">✦ RAG Assistant</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="brand-subtitle">'
        'Your document-grounded AI workspace'
        '</div>',
        unsafe_allow_html=True,
    )


    # -------------------------
    # Upload
    # -------------------------

    st.markdown(
        '<div class="section-title">'
        'Add Documents'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload PDF files",

        type=["pdf"],

        accept_multiple_files=True,

        label_visibility="collapsed",

        help="Add one or more PDFs to the knowledge base.",
    )


    if uploaded_files:

        if st.button(
            "Add to knowledge base",
            use_container_width=True,
            type="primary",
        ):

            progress = st.progress(
                0
            )

            added = 0

            for index, uploaded_file in enumerate(
                uploaded_files
            ):

                try:

                    add_pdf_to_database(
                        uploaded_file
                    )

                    added += 1

                    st.toast(
                        f"Added {uploaded_file.name}",
                        icon="✓",
                    )

                except Exception as exc:

                    st.error(
                        f"Could not add "
                        f"{uploaded_file.name}: "
                        f"{exc}"
                    )

                progress.progress(
                    (index + 1)
                    / len(uploaded_files)
                )


            if added:

                st.success(
                    f"{added} PDF"
                    f"{'s' if added != 1 else ''}"
                    " added successfully."
                )


    # -------------------------
    # How it works
    # -------------------------

    st.markdown(
        '<div class="section-title">'
        'How It Works'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "01  Upload a PDF"
    )

    st.caption(
        "02  Split & embed"
    )

    st.caption(
        "03  Ask questions"
    )


    st.divider()


    # -------------------------
    # Clear chat
    # -------------------------

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# MAIN UI
# =========================================================

st.markdown(
    '<div class="hero">',
    unsafe_allow_html=True,
)

st.title("Ask your documents.")

st.markdown(
    "Search your PDF knowledge base and get grounded answers."
)

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "View retrieved sources"
            ):

                for source in message["sources"]:

                    page = source.get(
                        "page"
                    )

                    name = source.get(
                        "source",
                        "Document",
                    )

                    if page is not None:

                        label = (
                            f"{name} · "
                            f"page {page + 1}"
                        )

                    else:

                        label = name


                    st.markdown(
                        f'<span class="source-pill">'
                        f'{label}'
                        f'</span>',
                        unsafe_allow_html=True,
                    )


# =========================================================
# CHAT INPUT
# =========================================================

query = st.chat_input(
    "Ask a question about your documents…"
)


if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            query
        )


    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching your documents…"
        ):

            try:

                answer, docs = answer_question(
                    query
                )

                st.markdown(
                    answer
                )


                # -------------------------
                # Sources
                # -------------------------

                sources = []

                for doc in docs:

                    sources.append(
                        {
                            "source": Path(
                                doc.metadata.get(
                                    "source",
                                    "Document",
                                )
                            ).name,

                            "page": doc.metadata.get(
                                "page"
                            ),
                        }
                    )


                if sources:

                    with st.expander(
                        "View retrieved sources"
                    ):

                        for source in sources:

                            page = source.get(
                                "page"
                            )

                            if page is not None:

                                label = (
                                    f'{source["source"]}'
                                    f' · page {page + 1}'
                                )

                            else:

                                label = source[
                                    "source"
                                ]


                            st.markdown(
                                f'<span class="source-pill">'
                                f'{label}'
                                f'</span>',
                                unsafe_allow_html=True,
                            )


                # -------------------------
                # Save response
                # -------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",

                        "content": answer,

                        "sources": sources,
                    }
                )


            except Exception as exc:

                error_message = (
                    "I couldn't process that request. "
                    "Please make sure your Mistral API key "
                    "is configured and the knowledge base "
                    "is available."
                )

                st.error(
                    error_message
                )

                with st.expander(
                    "Technical details"
                ):

                    st.code(
                        str(exc)
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",

                        "content": error_message,
                    }
                )