# Contract-Sentry NLP

Contract-Sentry is a domain-specific Text Analysis and Retrieval System focused on Indian Employment and Work-Related Contracts. It features a beautiful web GUI paired with a powerful Python NLP backend.

## 🚀 How to Run the Application

To run the application locally on your computer, you need to start two things: the **Python NLP Backend** and the **Frontend Web Server**.

### 1. Install Dependencies
Before running the app for the first time, make sure you have Python installed, then install all the required NLP and server libraries by running this command in your terminal:

```bash
pip install flask flask-cors spacy nltk pandas pymupdf tokenizers scikit-learn python-docx rapidocr onnxruntime pillow bs4
```

### 2. Start the Backend API Server
The backend handles the NLP pipeline (OCR, tokenization, POS tagging, and search). Open a terminal in the project folder and run:

```bash
python server.py
```
*(Note: It may take 15-30 seconds to load the AI models into memory. Wait until you see the success message saying it is running on port 5000).*

### 3. Start the Frontend Web Server
Open a **second, separate terminal** in the project folder and run:

```bash
python -m http.server 8000
```

### 4. Access the App
Once both servers are running, open your web browser (Chrome, Edge, etc.) and go to:
👉 **http://localhost:8000**

---

### Features Included (Tier 2):
- **Live NLP Pipeline Analysis**: Type custom sentences and see how they are tokenized, lemmatized, and POS-tagged live!
- **Custom Query Retrieval**: Type any keyword or boolean query into the search bar to query the contract inverted index.
- **Document Processing Upload**: UI support for dragging and dropping new `.pdf`, `.docx`, or image files into the system.
