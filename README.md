# TeluguGPT
LLM's output in Telugu
Here’s a sample `README.md` file for your **Telugu RAG Streamlit App**:

---

# **Telugu AI Assistant (Telugu RAG)**

This is a Retrieval-Augmented Generation (RAG) based AI assistant that processes queries in **Telugu** and provides intelligent responses using **Groq AI** and various NLP technologies.

---

## **Features**

- Accepts queries in the **Telugu language**.
- Uses a **retrieval-augmented generation** (RAG) approach to provide accurate and context-aware answers.
- Powered by:
  - **Hugging Face Datasets**: For dataset processing.
  - **FAISS Index**: For efficient similarity search.
  - **Groq API**: For generating answers using state-of-the-art models.
  - **Streamlit**: For a clean and interactive user interface.

---

## **How It Works**

1. **Dataset Processing:**
   - Telugu text is preprocessed and split into smaller chunks.
   - Sentence embeddings are generated using a multilingual transformer model.
   - FAISS index is built to retrieve the most relevant chunks for a given query.

2. **Query Flow:**
   - A user submits a question in Telugu through the Streamlit app.
   - Relevant chunks are retrieved from the FAISS index based on similarity.
   - The query and retrieved context are sent to Groq API for generating a response.

3. **Answer Generation:**
   - The Groq API generates a response in Telugu using its powerful language models.
   - The response is displayed in the app for the user.

---

## **Technology Stack**

- **Python**: Programming language.
- **Streamlit**: For creating an interactive web application.
- **Groq API**: To generate AI-powered responses.
- **Hugging Face Transformers**: For tokenization and embedding generation.
- **FAISS**: For similarity-based retrieval.
- **Sentence Transformers**: For generating chunk embeddings.

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/Telugu-RAG.git
cd Telugu-RAG
```

### **2. Install Dependencies**
- Create a virtual environment (optional but recommended):
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```
- Install required packages:
  ```bash
  pip install -r requirements.txt
  ```

### **3. Set Up Secrets**
- Add your Groq API key in a `secrets.toml` file:
  ```bash
  mkdir .streamlit
  echo "[groq]" > .streamlit/secrets.toml
  echo "api_key = 'your_groq_api_key'" >> .streamlit/secrets.toml
  ```

### **4. Run the App Locally**
```bash
streamlit run app.py
```

### **5. Deploy the App**
- Follow [Streamlit Cloud](https://streamlit.io/cloud) deployment instructions.

---

## **Demo**

You can access the deployed app at:  
[Live Demo](https://your-username-telugu-rag.streamlit.app)

---

## **File Structure**

```
RAG/
├── venv/                # Virtual environment (optional)
├── .streamlit/          # Streamlit secrets directory
│   └── secrets.toml     # Stores the Groq API key
├── app.py               # Streamlit app script
├── telugurag.py         # Dataset processing and FAISS index building
├── telugu_chunks.txt    # Preprocessed Telugu text chunks
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## **Future Improvements**

- Add support for other Indian languages.
- Enhance the retrieval pipeline for better context matching.
- Optimize response generation using custom fine-tuned models.

---

## **Contributions**

Contributions are welcome! Please open an issue or create a pull request for any suggestions or improvements.

---

## **License**

This project is licensed under the [MIT License](LICENSE).

---

Let me know if you'd like to make any additional tweaks or customize this content further! 🚀
