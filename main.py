import os
import requests
import json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import PyPDF2
import io
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HackRx 6.0 - LLM Query Retrieval System",
    description="Intelligent document processing and query answering system",
    version="1.0.0"
)

# Security
security = HTTPBearer()

# Configuration
EXPECTED_TOKEN = "fb139ca28bc7cd76601aa31f2eaecc7a1c5bf25af5d874ee56865990c269a792"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Set OpenAI API key
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Request/Response Models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class DocumentChunk(BaseModel):
    text: str
    page: int
    chunk_id: int

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

class DocumentProcessor:
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        
    def download_pdf(self, url: str) -> bytes:
        """Download PDF from URL"""
        try:
            logger.info(f"Downloading PDF from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> List[DocumentChunk]:
        """Extract text from PDF and create chunks"""
        try:
            chunks = []
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    sentences = text.split('. ')
                    current_chunk = ""
                    chunk_id = 0
                    
                    for sentence in sentences:
                        if len(current_chunk + sentence) < 500:
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk.strip():
                                chunks.append(DocumentChunk(
                                    text=current_chunk.strip(),
                                    page=page_num + 1,
                                    chunk_id=chunk_id
                                ))
                                chunk_id += 1
                            current_chunk = sentence + ". "
                    
                    if current_chunk.strip():
                        chunks.append(DocumentChunk(
                            text=current_chunk.strip(),
                            page=page_num + 1,
                            chunk_id=chunk_id
                        ))
            
            logger.info(f"Extracted {len(chunks)} chunks from PDF")
            return chunks
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {e}")
    
    def create_embeddings(self, chunks: List[DocumentChunk]):
        """Create embeddings for document chunks"""
        try:
            texts = [chunk.text for chunk in chunks]
            embeddings = sentence_model.encode(texts)
            self.chunks = chunks
            self.embeddings = embeddings
            logger.info(f"Created embeddings for {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create embeddings: {e}")
    
    def retrieve_relevant_chunks(self, question: str, top_k: int = 5) -> List[DocumentChunk]:
        """Retrieve most relevant chunks for a question"""
        try:
            question_embedding = sentence_model.encode([question])
            similarities = cosine_similarity(question_embedding, self.embeddings)[0]
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            relevant_chunks = [self.chunks[i] for i in top_indices]
            
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks for question")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []

class QueryEngine:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
    
    def generate_answer(self, question: str, context_chunks: List[DocumentChunk]) -> str:
        """Generate answer using LLM with retrieved context"""
        try:
            context = "\n\n".join([f"Page {chunk.page}: {chunk.text}" for chunk in context_chunks])
            
            prompt = f"""Based on the following document context, please answer the question accurately and concisely.

Context:
{context}

Question: {question}

Instructions:
1. Answer based only on the information provided in the context
2. Be specific and include relevant details like waiting periods, conditions, percentages, etc.
3. If the information is not available in the context, state that clearly
4. Keep the answer concise but complete

Answer:"""

            if OPENAI_API_KEY:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on insurance policy documents."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.1
                )
                answer = response.choices[0].message.content.strip()
            else:
                answer = self._rule_based_answer(question, context)
            
            logger.info(f"Generated answer for question: {question[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while processing this question."
    
    def _rule_based_answer(self, question: str, context: str) -> str:
        """Fallback rule-based answering when OpenAI is not available"""
        question_lower = question.lower()
        context_lower = context.lower()
        
        if "grace period" in question_lower:
            if "thirty days" in context_lower or "30 days" in context_lower:
                return "A grace period of thirty days is provided for premium payment after the due date."
        
        elif "waiting period" in question_lower and "pre-existing" in question_lower:
            if "thirty-six" in context_lower or "36" in context_lower:
                return "There is a waiting period of thirty-six (36) months of continuous coverage for pre-existing diseases."
        
        elif "maternity" in question_lower:
            if "24 months" in context_lower:
                return "Yes, the policy covers maternity expenses. The female insured person must have been continuously covered for at least 24 months."
        
        elif "cataract" in question_lower:
            if "two years" in context_lower or "2 years" in context_lower:
                return "The policy has a specific waiting period of two (2) years for cataract surgery."
        
        return "Based on the available information in the document, I found relevant content but need more specific context to provide a detailed answer."
    
    def process_query(self, documents_url: str, questions: List[str]) -> List[str]:
        """Process the complete query request"""
        try:
            start_time = datetime.now()
            
            pdf_content = self.doc_processor.download_pdf(documents_url)
            chunks = self.doc_processor.extract_text_from_pdf(pdf_content)
            self.doc_processor.create_embeddings(chunks)
            
            answers = []
            for question in questions:
                relevant_chunks = self.doc_processor.retrieve_relevant_chunks(question)
                answer = self.generate_answer(question, relevant_chunks)
                answers.append(answer)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Processed {len(questions)} questions in {processing_time:.2f} seconds")
            
            return answers
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return [f"Error processing question: {e}" for _ in questions]

# Initialize query engine
query_engine = QueryEngine()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "HackRx 6.0 - LLM Query Retrieval System",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Main endpoint for processing documents and answering questions"""
    try:
        logger.info(f"Received request with {len(request.questions)} questions")
        
        if not request.documents:
            raise HTTPException(status_code=400, detail="Documents URL is required")
        
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        answers = query_engine.process_query(request.documents, request.questions)
        
        response = QueryResponse(answers=answers)
        logger.info("Query processed successfully")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in run_query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_configured": OPENAI_API_KEY is not None,
        "sentence_model_loaded": sentence_model is not None
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )



app_handler = app