# Advanced PDF Chatbot

This project provides an advanced chatbot powered by machine learning, capable of processing PDF documents and answering questions based on their contents. It uses **Sentence Transformers** for embedding and **TinyLlama** for generating responses, along with **Gradio** for the user interface.

## Features

- **PDF Processing**: Upload PDF files to extract and embed text chunks from them.
- **Context Retrieval**: For a given query, the system retrieves the most relevant context from the PDF.
- **Question Answering**: The chatbot uses the retrieved context to generate a concise answer to the user's question based on the document's contents.
- **Gradio Interface**: A user-friendly web interface where users can upload PDFs, ask questions, and receive responses.

## Installation

### Step 1: Set Up a Python Virtual Environment

It is recommended to use a virtual environment to manage the dependencies for this project. To set up a virtual environment, run the following commands:

1. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment**:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

### Step 2: Install Dependencies

With the virtual environment activated, install the required dependencies from `requirements.txt`. Additionally, if you are running this in a server or production environment, install `uvicorn` to serve the application:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

```bash
python main.py
```
This will start a local development server, allowing you to interact with the chatbot via the Gradio interface at `http://127.0.0.1:7861` (can differ too).

## How to Use

1. **Upload PDF**: Click the "Upload PDF" button to upload a PDF document. The system will process the document and extract text from it.
   
2. **Ask a Question**: After the document is processed, enter your question in the chat box. The chatbot will retrieve the relevant context from the document and generate a response based on the content.

3. **Response**: The chatbot will provide a clear, concise answer, or inform you if the document does not contain enough information to answer the question.

## Customizing Response Generation and Model

- **Change the Response Attribute**: The model used for generating responses can be modified by adjusting the `tinyllama_api_url` in the code to point to another API or a different model endpoint.
  
- **Switch to a Different Model**: If you wish to use a different model for response generation, simply change the `tinyllama_api_url` to the appropriate model URL and update the `model` field in the API call within the `generate_response` method to the new model’s name.

Example:

```python
self.tinyllama_api_url = "http://localhost:12345/api/generate"  # Replace with your model's API URL
```

```python
"model": "new-model-name"  # Replace with the desired model name
```

These adjustments allow you to switch between different models or APIs as required.

## Requirements

- Python 3.7+
- CUDA (optional for GPU acceleration)

## Troubleshooting

- **TinyLlama API Not Found**: Ensure that the TinyLlama API is running locally on `http://localhost:11434`. The chatbot requires the TinyLlama API to generate responses.
  
- **CUDA Issues**: If you're using a GPU, ensure you have the proper CUDA version installed for PyTorch. If no GPU is available, the code will fall back to using CPU.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Requirements for `requirements.txt`

Ensure the following packages are included in the `requirements.txt` for the project:

```text
gradio
torch
sentence-transformers
numpy
pandas
requests
langchain
langchain_community
scikit-learn
pypdf
```
