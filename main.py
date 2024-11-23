import os
import re
import logging
import traceback
from typing import List, Dict, Any, Optional

import gradio as gr
import torch
import numpy as np
import pandas as pd
import requests

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

tinyllama_api_url="http://localhost:11434/api/generate"

class AdvancedPDFChatbot:
    def __init__(
        self, 
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k_chunks: int = 3,
    ):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        self.device = self._configure_device()

        self.logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_chunks = top_k_chunks
        self.tinyllama_api_url = tinyllama_api_url

        self.reset_document_store()

    def _configure_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def reset_document_store(self):
        self.document_chunks: List[str] = []
        self.document_embeddings: List[np.ndarray] = []
        self.document_metadata: List[Dict[str, Any]] = []

    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        try:
            self.reset_document_store()
            
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            texts = text_splitter.split_documents(documents)
            
            for idx, text in enumerate(texts):
                chunk = self.preprocess_text(text.page_content)
                
                if not chunk:
                    continue
                
                embedding = self.embedding_model.encode(chunk, convert_to_numpy=True)
                
                self.document_chunks.append(chunk)
                self.document_embeddings.append(embedding)
                self.document_metadata.append({
                    'source': pdf_path,
                    'page': text.metadata.get('page', 0),
                    'chunk_id': idx
                })
            
            result = {
                'total_chunks': len(self.document_chunks),
                'document_source': pdf_path
            }
            
            self.logger.info(f"PDF processed: {result}")
            return result
        
        except Exception as e:
            self.logger.error(f"PDF processing error: {e}")
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}

    def retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        try:
            processed_query = self.preprocess_text(query)
            query_embedding = self.embedding_model.encode(processed_query, convert_to_numpy=True)
            
            similarities = cosine_similarity(query_embedding.reshape(1, -1), 
                                             np.array(self.document_embeddings))[0]
            
            top_indices = np.argsort(similarities)[-self.top_k_chunks:][::-1]
            
            context_results = [
                {
                    'text': self.document_chunks[idx],
                    'similarity': similarities[idx],
                    'metadata': self.document_metadata[idx]
                }
                for idx in top_indices
            ]
            
            return context_results
        
        except Exception as e:
            self.logger.error(f"Context retrieval error: {e}")
            return []

    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        try:
            context_texts = [item['text'] for item in context]
            context_str = "\n".join(context_texts)
            
            prompt = f"""You are a helpful AI assistant answering questions based on a PDF document.

Context from document:
{context_str}

Question: {query}

Please provide a clear, concise answer based strictly on the given context. If the context does not contain sufficient information to answer the question, say "I cannot find the answer in the provided document."
"""
            
            response = requests.post(
                self.tinyllama_api_url,
                json={
                "model": "tinyllama:latest", "prompt": prompt,"stream": False
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                generated_text = response.json()['response']
                return generated_text
            else:
                self.logger.error(f"TinyLlama API error: {response.text}")
                return "I'm unable to generate a response at the moment."
        
        except Exception as e:
            self.logger.error(f"Response generation error: {e}")
            return "I'm unable to generate a response at the moment."

    def chat(self, query: str) -> str:
        context = self.retrieve_context(query)
        response = self.generate_response(query, context)
        return response

def create_gradio_interface():
    chatbot = AdvancedPDFChatbot()
    
    def process_pdf(file):
        result = chatbot.process_pdf(file.name)
        return str(result)
    
    def chat_interface(message, history):
        return chatbot.chat(message)
    
    with gr.Blocks() as demo:
        gr.Markdown("# Student Helper Chatbot")
        
        with gr.Row():
            pdf_input = gr.File(label="Upload PDF", type="filepath")
            process_btn = gr.Button("Process PDF")
        
        process_output = gr.Textbox(label="Processing Status")
        
        chatbot_interface = gr.ChatInterface(
            fn=chat_interface,
            title="Chat Here",
            description="Ask questions about your uploaded PDF",
            chatbot=gr.Chatbot(height=500)
        )
        
        process_btn.click(
            process_pdf, 
            inputs=pdf_input, 
            outputs=process_output
        )
    
    return demo

def main():
    demo = create_gradio_interface()
    demo.launch(
        debug=True
    )

if __name__ == "__main__":
    main()
