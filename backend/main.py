import os
import json
import re
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from tenacity import retry, stop_after_attempt, wait_exponential

# Set API (recommend using environment variables in production)
os.environ["OPENAI_API_KEY"] = "sk-ZcZiTXMC4NrbtDGdXl0Vu8WwPcVamZj0gGUsWoL3dGt0XXmG"
os.environ["OPENAI_API_BASE"] = "https://api.gptgod.online/v1"

# Global variables for resources
vector_store = None
llm = None
features_range_df = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load resources
    global vector_store, llm, features_range_df
    
    try:
        print("🚀 Loading FAISS vector database...")
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = FAISS.load_local(
            "/Users/zhangdi/Documents/workspace/langchain/pubmed_faiss_index_new",
            embeddings_model,
            allow_dangerous_deserialization=True
        )
        
        print("🤖 Initializing language model...")
        llm = init_chat_model("gemini-2.5-flash-all", model_provider="openai")
        
        print("📊 Loading feature reference data...")
        features_range_df = pd.read_csv("selected_features/test2&train_HC_avg_normalrange.csv", index_col=0)
        features_range_df = features_range_df.transpose()
        features_range_df.index.name = 'feature_name'
        features_range_df.reset_index(inplace=True)
        
        print("✅ All resources loaded successfully! Service is ready.")
        
    except Exception as e:
        print(f"❌ Resource loading failed: {e}")
        raise e
    
    yield  # Application runs here
    
    # Shutdown: clean up resources (optional)
    print("🛑 Shutting down service...")

app = FastAPI(
    title="Alzheimer's Disease Risk Assessment API",
    description="Personalized risk assessment service based on medical literature",
    version="1.0.0",
    lifespan=lifespan  # Use the new lifespan handler
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request data model
class PatientAssessmentRequest(BaseModel):
    subject_id: str
    age: int
    sex: str
    mmse: Optional[float] = None
    feature_values: Dict[str, float]

class AssessmentResponse(BaseModel):
    subject_id: str
    risk_level: str
    risk_score: Optional[float] = None
    explanation: str
    cited_literature: List[str]
    status: str = "success"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def get_embeddings(text, embeddings_model):
    embedding = embeddings_model.embed_query(text)
    if not embedding or not isinstance(embedding, list):
        raise ValueError("Invalid embedding")
    return embedding

def format_number(value):
    try:
        num = float(value)
        if num == 0:
            return "0.00"
        
        if 1e-5 <= abs(num) < 0.01:
            sci_str = f"{num:.4e}"
            coeff, exp = sci_str.split('e')
            exp = int(exp)
            coeff_rounded = f"{float(coeff):.2f}"
            return f"{coeff_rounded} * 10^({exp})"
        
        return f"{num:.2f}"
    except (ValueError, TypeError):
        return "N/A"

def get_feature_reference(feature_name: str, value: float) -> str:
    global features_range_df
    
    if features_range_df is None:
        return f"(No reference data available)"
    
    feature_row = features_range_df[features_range_df['feature_name'] == feature_name]
    if not feature_row.empty:
        try:
            avg = format_number(feature_row['avg'].values[0])
            p5 = format_number(feature_row['5th percentile'].values[0])
            p95 = format_number(feature_row['95th percentile'].values[0])
            return f"(HC avg: {avg}, normal range: {p5}-{p95})"
        except:
            return f"(Reference data format error)"
    
    return f"(No HC reference data)"

@app.get("/")
async def root():
    return {
        "message": "Alzheimer's Disease Risk Assessment API Service is running",
        "version": "1.0.0",
        "endpoints": {
            "assessment": "/assess-risk/ (POST)",
            "docs": "/docs"
        }
    }

@app.get("/features-range")
async def get_features_range():
    if features_range_df is None:
        raise HTTPException(status_code=500, detail="Feature data not loaded")
    
    features = features_range_df['feature_name'].tolist()
    return {"available_features": features}

@app.post("/assess-risk/", response_model=AssessmentResponse)
async def assess_risk(request: PatientAssessmentRequest):
    try:
        if vector_store is None or llm is None:
            raise HTTPException(status_code=500, detail="System resources not ready")
        
        print(f"🔍 Processing assessment request for patient {request.subject_id}...")
        
        brain_volume_lines = []
        white_matter_lines = []
        
        for feature_name, value in request.feature_values.items():
            formatted_value = format_number(value)
            reference_info = get_feature_reference(feature_name, value)
            
            feature_line = f"- **{feature_name}**: {formatted_value} {reference_info}"
            
            wm_keywords = ["fractional_anisotropy", "mean_diffusivity", "axial_diffusivity", "radial_diffusivity"]
            if any(kw in feature_name.lower() for kw in wm_keywords):
                white_matter_lines.append(feature_line)
            else:
                brain_volume_lines.append(feature_line)
        
        query = (
            f"Alzheimer's disease risk assessment. Patient: {request.age} years old, {request.sex}, "
            f"MMSE: {request.mmse if request.mmse else 'N/A'}. "
            f"Key features: {list(request.feature_values.keys())}"
        )
        
        try:
            embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
            emb = get_embeddings(query, embeddings_model)
            docs = vector_store.similarity_search_by_vector(emb, k=3)
            cited_literature = [
                f"PMID: {doc.metadata.get('pmid', 'Unknown')}, year: {doc.metadata.get('year')}" 
                for doc in docs
            ]
        except Exception as e:
            print(f"Literature retrieval failed: {e}")
            cited_literature = []
            docs = []
        
        messages = [
            SystemMessage(content="""You are a professional neurologist. Based on the provided patient data and relevant medical literature, 
                            provide a professional, rigorous analysis of the patient's Alzheimer's disease risk. Use inline citation format (PMID: XXXX) to reference literature."""),
            SystemMessage(content="""You are a neurologist specializing in dementia. 
Provide a concise clinical analysis in 3-4 sentences. Focus on:
1. Key abnormal features
2. Clinical significance
3. Recommended next steps
Use professional but clear language."""),
            HumanMessage(content=f"Patient基本信息: Age {request.age}, Sex {request.sex}, MMSE score {request.mmse if request.mmse else 'Not provided'}."),
            
            HumanMessage(content="Brain volume features:\n" + "\n".join(brain_volume_lines) if brain_volume_lines else "No brain volume feature data"),
            
            HumanMessage(content="White matter microstructure features:\n" + "\n".join(white_matter_lines) if white_matter_lines else "No white matter feature data"),
            
            HumanMessage(content="Relevant medical literature abstracts:\n" + "\n".join([
                f"{doc.page_content[:300]}... (PMID: {doc.metadata.get('pmid', 'Unknown')})" 
                for doc in docs
            ]) if docs else "No relevant literature retrieved"),
            
            HumanMessage(content="""Please provide a detailed risk analysis report:
1. First analyze the changes in brain volume features and their clinical significance
2. Then analyze the alterations in white matter microstructure features  
3. Provide a comprehensive risk assessment conclusion based on all information
4. Be sure to cite relevant literature to support your analysis
Provide the analysis results directly, without additional introductions or conclusions."""),
        ]
        
        try:
            response = llm.invoke(messages)
            explanation = response.content.strip()
        except Exception as e:
            explanation = f"Risk assessment generation failed: {str(e)}"
        
        risk_score = calculate_risk_score(request)
        risk_level = "High" if risk_score > 0.7 else "Moderate" if risk_score > 0.4 else "Low"
        
        return AssessmentResponse(
            subject_id=request.subject_id,
            risk_level=risk_level,
            risk_score=risk_score,
            explanation=explanation,
            cited_literature=cited_literature
        )
        
    except Exception as e:
        print(f"Assessment process error: {e}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

def calculate_risk_score(request: PatientAssessmentRequest) -> float:
    base_score = 0.3
    
    if request.age > 75:
        base_score += 0.3
    elif request.age > 65:
        base_score += 0.2
    elif request.age > 55:
        base_score += 0.1
    
    if request.mmse and request.mmse < 24:
        base_score += 0.2
    
    for feature_name, value in request.feature_values.items():
        if "volume" in feature_name.lower() and value < 0.8:
            base_score += 0.1
        if "diffusivity" in feature_name.lower() and value > 1.2:
            base_score += 0.1
    
    return min(1.0, base_score)

if __name__ == "__main__":
    import uvicorn
    # Remove reload=True or use import string format
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)