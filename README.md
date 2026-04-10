# 🚀 RAG Practice: Document Summarizer

Welcome to the **RAG Practice** repository! This project demonstrates how to leverage **LangChain** and **Mistral AI** to summarize and analyze various forms of documents, such as Text and PDFs. 

Currently, the project uses the `mistral-small-2603` model to interpret the contents of provided files and return a concise, intelligent summary.

---

## 🌟 Features

- 📄 **Text Loading:** Extracts and summarizes raw text from `.txt` files using LangChain's `TextLoader`.
- 📕 **PDF Loading:** Extracts text from `.pdf` files and summarizes the content using `PyPDFLoader`.
- 🌐 **Web Loading:** Scrapes and summarizes web pages from URLs using `WebBaseLoader`.
- 🧠 **LLM Integration:** Powered by **Mistral AI** to generate smart, contextual, and accurate summaries.
- ⚙️ **Prompt Templating:** Utilizes LangChain's `ChatPromptTemplate` to seamlessly structure instructions and context for the AI.

---

## 📂 Project Structure

```text
RAG_Practice/
├── main.py                     # Script to summarize plain text files
├── pdfloader.py                # Script to summarize PDF files
├── webloader.py                # Script to summarize web pages from URLs
├── document_loaders/
│   ├── notes.txt               # Sample text file (Machine Learning notes)
│   ├── ml.pdf                  # Sample PDF file
│   └── test.py                 # Sandbox for testing loaders
└── .env                        # Environment variables (API Keys!)
```

---

## 🛠️ Getting Started

Follow these instructions to set up the project locally on your machine.

### 1. Navigate to the Directory
Assuming you already have the folder, open your terminal and change into the project directory:
```bash
cd /Users/arhanalam/Desktop/RAG_Practice
```

### 2. Create a Virtual Environment
It's always best practice to keep your Python dependencies isolated per project to prevent conflicts. Let's create a virtual environment named `.venv`:
```bash
python3 -m venv .venv
```

*(Alternatively, if you are using `uv`, you can run `uv venv`)*

### 3. Activate the Virtual Environment
Before installing dependencies or running the scripts, you **must** activate the environment.

**On macOS / Linux:**
```bash
source .venv/bin/activate
```
*(You will know it's successfully activated when you see `(.venv)` at the beginning of your terminal prompt!)*

### 4. Install Dependencies
Make sure you install the necessary packages using `pip` (or `uv`):
```bash
pip install langchain langchain_mistralai langchain_community python-dotenv pypdf beautifulsoup4
```

### 5. Setup Environment Variables
This project requires API keys to communicate with Mistral AI.
Create a `.env` file in the root of the project if you haven't already:
```bash
touch .env
```
Open the `.env` file and securely add your Mistral API key:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

---

## 🚀 Usage

Make sure you are in the root directory `RAG_Practice` and your virtual environment is **activated** before running the scripts!

### 📝 Summarize a Text File
Run `main.py`. This will load the text from `document_loaders/notes.txt`, inject it into the AI prompt, and print out the summary:
```bash
python main.py
```

### 📚 Summarize a PDF File
Run `pdfloader.py`. This reads the first page of `document_loaders/ml.pdf` and provides a comprehensive summary mapping out the study!
```bash
python pdfloader.py
```

### 🌐 Summarize a Web Page
Run `webloader.py`. This fetches the page content from a specified URL (e.g., Wikipedia) and generates a structured summary of the article:
```bash
python webloader.py
```

---

## 💡 Pro-tips & Gotchas

*   **Paths Matter:** When specifying a file in the loader (e.g., `TextLoader("document_loaders/notes.txt")`), the path is evaluated **relative to your Current Working Directory** where you ran the `python` command, not relative to the script itself. Always run your scripts from the root `RAG_Practice` folder.
*   **Deactivate:** Once you are done coding for the day, you can cleanly exit the virtual environment by typing:
    ```bash
    deactivate
    ```

Happy Coding! 💻✨
