"""RAG System for Clinical Notes Retrieval and Summarization.

This module implements a Retrieval-Augmented Generation system for
processing clinical notes and generating patient context summaries.
"""

import os
import logging
from typing import List, Dict
import pandas as pd
import numpy as np
from snowflake.snowpark import Session
from snowflake.cortex import Complete, ExtractAnswer, Sentiment, Summarize, Translate
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalNotesRAG:
    """RAG system for clinical notes retrieval and summarization."""
    
    def __init__(self, snowflake_session: Session, embedding_model: str = 'all-MiniLM-L6-v2'):
        """Initialize the RAG system.
        
        Args:
            snowflake_session: Active Snowflake session
            embedding_model: Name of the sentence transformer model
        """
        self.session = snowflake_session
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB for vector storage
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        
        self.collection_name = "clinical_notes"
        self.collection = None
        
    def initialize_vector_store(self):
        """Initialize or load the vector store collection."""
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def load_clinical_notes(self, patient_id: str = None) -> pd.DataFrame:
        """Load clinical notes from Snowflake.
        
        Args:
            patient_id: Optional patient ID to filter notes
            
        Returns:
            DataFrame with clinical notes
        """
        where_clause = f"WHERE patient_id = '{patient_id}'" if patient_id else ""
        
        query = f"""
        SELECT 
            note_id,
            patient_id,
            encounter_id,
            note_date,
            note_type,
            author_role,
            note_text
        FROM clinical_notes
        {where_clause}
        ORDER BY note_date DESC
        """
        
        df = self.session.sql(query).to_pandas()
        logger.info(f"Loaded {len(df)} clinical notes")
        
        return df
    
    def embed_notes(self, notes_df: pd.DataFrame):
        """Embed clinical notes and store in vector database.
        
        Args:
            notes_df: DataFrame with clinical notes
        """
        if self.collection is None:
            self.initialize_vector_store()
        
        logger.info("Embedding clinical notes...")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            notes_df['note_text'].tolist(),
            show_progress_bar=True
        )
        
        # Prepare documents for ChromaDB
        documents = notes_df['note_text'].tolist()
        ids = notes_df['note_id'].tolist()
        metadatas = notes_df[[
            'patient_id', 'encounter_id', 'note_date', 
            'note_type', 'author_role'
        ]].to_dict('records')
        
        # Convert datetime to string for metadata
        for metadata in metadatas:
            if pd.notna(metadata['note_date']):
                metadata['note_date'] = str(metadata['note_date'])
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        
        logger.info(f"Embedded and stored {len(documents)} notes")
    
    def retrieve_relevant_notes(self, 
                               query: str, 
                               patient_id: str = None,
                               n_results: int = 5) -> List[Dict]:
        """Retrieve relevant clinical notes for a query.
        
        Args:
            query: Search query
            patient_id: Optional patient ID to filter results
            n_results: Number of results to return
            
        Returns:
            List of relevant notes with metadata
        """
        if self.collection is None:
            self.initialize_vector_store()
        
        # Embed query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Build where filter
        where_filter = {"patient_id": patient_id} if patient_id else None
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_filter
        )
        
        # Format results
        relevant_notes = []
        for i in range(len(results['ids'][0])):
            relevant_notes.append({
                'note_id': results['ids'][0][i],
                'note_text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        logger.info(f"Retrieved {len(relevant_notes)} relevant notes")
        return relevant_notes
    
    def generate_patient_summary(self, patient_id: str) -> Dict:
        """Generate a comprehensive summary for a patient using RAG.
        
        Args:
            patient_id: Patient ID
            
        Returns:
            Dictionary with summary and source notes
        """
        # Retrieve relevant notes
        query = """Patient history including housing status, social determinants of health, 
        psychiatric conditions, substance use, emergency visits, and discharge planning"""
        
        relevant_notes = self.retrieve_relevant_notes(
            query=query,
            patient_id=patient_id,
            n_results=10
        )
        
        if not relevant_notes:
            return {
                'patient_id': patient_id,
                'summary': 'No clinical notes available for this patient.',
                'source_note_ids': []
            }
        
        # Concatenate relevant notes
        combined_text = "\n\n".join([
            f"Note Type: {note['metadata']['note_type']}\n" +
            f"Date: {note['metadata']['note_date']}\n" +
            f"Content: {note['note_text']}"
            for note in relevant_notes
        ])
        
        # Generate summary using Snowflake Cortex
        prompt = f"""Based on the following clinical notes, generate a concise summary 
focusing on:
1. Housing status and homelessness risk factors
2. Social determinants of health
3. Psychiatric and substance use history
4. Recent emergency department visits
5. Discharge planning and follow-up needs

Clinical Notes:
{combined_text}

Provide a structured summary highlighting key risk factors:"""
        
        summary_response = Complete(
            model='mistral-large',
            prompt=prompt,
            session=self.session
        )
        
        summary_text = summary_response
        
        # Store summary in Snowflake
        source_note_ids = [note['note_id'] for note in relevant_notes]
        
        result = {
            'patient_id': patient_id,
            'summary': summary_text,
            'source_note_ids': source_note_ids
        }
        
        logger.info(f"Generated summary for patient {patient_id}")
        return result
    
    def save_summary_to_snowflake(self, summary: Dict, model_version: str):
        """Save RAG summary to Snowflake table.
        
        Args:
            summary: Summary dictionary
            model_version: Model version identifier
        """
        summary_data = pd.DataFrame([{
            'patient_id': summary['patient_id'],
            'summary_text': summary['summary'],
            'source_note_ids': summary['source_note_ids'],
            'model_version': model_version,
            'generated_at': pd.Timestamp.now()
        }])
        
        # Write to Snowflake
        snow_df = self.session.create_dataframe(summary_data)
        snow_df.write.mode('append').save_as_table('rag_summaries')
        
        logger.info(f"Saved summary for patient {summary['patient_id']} to rag_summaries table")
    
    def batch_generate_summaries(self, patient_ids: List[str] = None) -> List[Dict]:
        """Generate summaries for multiple patients.
        
        Args:
            patient_ids: List of patient IDs. If None, process all patients with notes.
            
        Returns:
            List of summary dictionaries
        """
        if patient_ids is None:
            # Get all patients with clinical notes
            query = "SELECT DISTINCT patient_id FROM clinical_notes"
            patient_ids = self.session.sql(query).to_pandas()['patient_id'].tolist()
        
        summaries = []
        for patient_id in patient_ids:
            try:
                summary = self.generate_patient_summary(patient_id)
                summaries.append(summary)
                self.save_summary_to_snowflake(summary, model_version="v1.0")
            except Exception as e:
                logger.error(f"Error generating summary for patient {patient_id}: {e}")
        
        logger.info(f"Generated {len(summaries)} summaries")
        return summaries


if __name__ == "__main__":
    from ml_pipeline import create_snowflake_session
    
    # Create session
    session = create_snowflake_session()
    
    # Initialize RAG system
    rag = ClinicalNotesRAG(session)
    rag.initialize_vector_store()
    
    # Load and embed all clinical notes
    notes_df = rag.load_clinical_notes()
    rag.embed_notes(notes_df)
    
    # Generate summaries for all patients
    summaries = rag.batch_generate_summaries()
    
    print(f"Generated {len(summaries)} patient summaries")
    
    session.close()
