import streamlit as st
from rag import process_urls, generate_answer, initialize_components

st.set_page_config(page_title="Real Estate Research Tool", page_icon="🏙️")
st.title("🏙️ Real Estate Research Tool")

# Sidebar Inputs
st.sidebar.header("URL Input")
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

placeholder = st.empty()

# Initialize session state variables
if "urls_processed" not in st.session_state:
    st.session_state["urls_processed"] = False

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

process_url_button = st.sidebar.button("Process URLs")

# --- URL Processing Logic ---
if process_url_button:
    urls = [url for url in (url1, url2, url3) if url.strip() != ""]
    if len(urls) == 0:
        placeholder.error("⚠️ You must provide at least one valid URL.")
    else:
        placeholder.info("Processing URLs... please wait.")
        for status in process_urls(urls):
            placeholder.text(status)
        st.session_state["urls_processed"] = True
        st.success("✅ URLs processed successfully! You can now ask questions.")

# --- Disable Chat Before URLs Are Processed ---
if not st.session_state["urls_processed"]:
    st.info("Please process at least one URL before asking questions.")
    st.stop()

# --- Question Input and Answer Display ---
query = st.text_input("💬 Ask a question:")
if query:
    try:
        answer, sources = generate_answer(query)
        st.header("Answer:")
        st.write(answer)

        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
    except RuntimeError as e:
        st.error("⚠️ You must process URLs first before chatting.")
