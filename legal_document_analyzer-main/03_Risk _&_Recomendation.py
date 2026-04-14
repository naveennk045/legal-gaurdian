from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY= os.environ.get("Api_key")
    

def initialize_llm(api_key, model_name="llama-3.2-11b-vision-preview"):
    """Initialize the language model and embeddings"""
    llm = ChatGroq(groq_api_key=api_key, model_name=model_name)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return llm, embeddings

def process_document(file_path, embeddings):
    """Process a document file and create vector store"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=20
        )
        texts = text_splitter.split_text(content)
        
        # Create vector store
        vector_store = Chroma.from_texts(texts, embeddings)
        return texts, vector_store
    
    except Exception as e:
        raise Exception(f"Error processing document: {str(e)}")

def analyze_risks(context, llm, vector_store):
    """Analyze risks in the given context"""
    prompt = f"""
    Analyze the following text for legal and business risks. Provide:
    1. Key risks identified
    2. Specific recommendations to address each risk
    
    Text: {context}
    """
    
    retriever = vector_store.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    response = qa.invoke(prompt)
    
    return response["result"]

def generate_recommendations(texts, llm, vector_store):
    """Generate recommendations from analyzed texts"""
    results = []
    
    for chunk in texts:
        analysis = analyze_risks(chunk, llm, vector_store)
        
        # Parse the analysis into structured format
        result = {
            "context": chunk,
            "risks": analysis.split("Recommendations:")[0].strip(),
            "recommendations": analysis.split("Recommendations:")[-1].strip()
            if "Recommendations:" in analysis
            else "No specific recommendations provided"
        }
        results.append(result)
    
    return results

def export_to_csv(results, output_path):
    """Export results to CSV file"""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    return output_path

def process_and_analyze(file_path, api_key=API_KEY, output_path="risk_analysis.csv"):
    """Main function to process document and generate recommendations"""
    try:
        # Initialize models
        llm, embeddings = initialize_llm(api_key)
        
        # Process document
        texts, vector_store = process_document(file_path, embeddings)
        
        # Generate recommendations
        results = generate_recommendations(texts, llm, vector_store)
        
        # Export to CSV
        csv_path = export_to_csv(results, output_path)
        
        return results, csv_path
    
    except Exception as e:
        raise Exception(f"Error in recommendation processing: {str(e)}")






"""   
FILE_PATH = "path_to_your_document.txt"

results, csv_path = main(FILE_PATH, API_KEY=API_KEY)
print(f"Analysis completed. Results saved to: {csv_path}")
"""
