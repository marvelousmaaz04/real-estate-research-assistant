# 🏙️ Real Estate Research Tool

A user-friendly, AI-powered research tool designed for effortless information retrieval and analysis of real estate news articles.
Users can input article URLs, process them into a vector database, and then ask natural language questions to receive contextually accurate insights — complete with cited sources.
*(While this project is built for the real estate domain, it can easily be extended to any domain.)*

-----

### 🚀 Features

  - **Fetch and process article content:** Load and extract text from online news articles using `AsyncHtmlLoader` and `Html2TextTransformer`.
  - **Semantic chunking and vector storage:** Automatically split text into meaningful chunks and store embeddings in **ChromaDB**.
  - **Advanced embeddings:** Uses **HuggingFace Sentence Transformers** (`all-MiniLM-L6-v2`) for efficient semantic similarity search.
  - **LLM-powered Q\&A:** Query articles via **Llama 3.3 70B (Groq)** for accurate answers with cited sources.
  - **Streamlit Interface:** Simple and interactive UI for URL input, processing, and question answering.

-----

## ⚙️ Setup

### 🧩 Option A — Using `pip`

1.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Create a `.env` file**
    In the project root, add your Groq API key:
    ```bash
    GROQ_API_KEY=your_groq_api_key
    ```
3.  **Run the Streamlit app**
    ```bash
    streamlit run main.py
    ```

-----

### ⚡ Option B — Using `uv` (Recommended)

[`uv`](https://www.google.com/search?q=%5Bhttps://github.com/astral-sh/uv%5D\(https://github.com/astral-sh/uv\)) is a modern, ultra-fast Python package manager that automatically handles virtual environments and dependency resolution.

1.  **Install dependencies**
    ```bash
    uv sync
    ```
    *(This will automatically create a virtual environment and install everything from `pyproject.toml`.)*
2.  **Create a `.env` file**
    In the project root, add your Groq API key:
    ```bash
    GROQ_API_KEY=your_groq_api_key
    ```
3.  **Run the Streamlit app**
    ```bash
    streamlit run main.py
    ```

-----

### 🧩 How It Works

1.  Enter one or more URLs in the sidebar.
2.  Click “Process URLs” — this will:
      * Load and clean the article text.
      * Split it into chunks.
      * Generate vector embeddings.
      * Store them in a persistent ChromaDB collection.
3.  Once processing is done, type a question related to those articles in the main input box.
4.  The app retrieves the most relevant chunks, queries the Groq Llama 3.3 model, and returns:
      * A concise, context-aware answer.
      * A list of source URLs supporting that answer.

-----

### 🧠 Example Usage

After setup, open the Streamlit app in your browser.

**Example URLs**

  * How the Federal Reserve’s rate policy affects mortgages
  * Why mortgage rates jumped despite Fed interest rate cut
  * Wall Street sees upside in 2025 for dividend-paying real estate stocks

**Example Query**

> What was the 30-year fixed mortgage rate mentioned, and when was it recorded?

**Example Output:**

**Answer:**
The 30-year fixed mortgage rate was around 6.6% as of December 2024.

**Sources:**

  * [https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html](https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html)

-----

### 🧰 Tech Stack

| Component | Description |
| --- | --- |
| Streamlit | UI framework for the interactive web app |
| LangChain | Orchestration framework for document processing and retrieval |
| AsyncHtmlLoader | HTML content loader compatible with Streamlit Cloud |
| Html2TextTransformer | Converts HTML into readable plain text |
| HuggingFace Embeddings | Embedding model for semantic search |
| ChromaDB | Vector database for document storage and retrieval |
| ChatGroq (Llama 3.3) | Large language model for Q\&A generation |

-----

### 📦 Project Structure

```
.
├── main.py             # Streamlit app entry point
├── rag.py              # Core logic for processing and querying
├── resources/
│   ├── image.png       # Screenshot for README
│   └── vectorstore/    # Persisted ChromaDB storage
├── requirements.txt
└── .env
```

-----

### 📘 Notes

  * The app uses `AsyncHtmlLoader`, which works seamlessly on Streamlit Cloud (no browser dependencies).
  * For local use, you can optionally replace it with `PlaywrightURLLoader` for full JavaScript-rendered content extraction.
  * The project currently uses Groq’s `Llama 3.3 70B Versatile` model for fast, high-quality answers.
  * Extendable to other domains such as finance, technology, healthcare, or education by simply changing the article URLs.