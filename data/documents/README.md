This folder is reserved for private documents used by the project's RAG pipeline.
Place local PDF documents here when they need to be ingested into the configured Pinecone index.
Use ingest_rag.py to extract text, create chunks, generate embeddings, and upload the document data to Pinecone.
Private documents should not be committed to the repository.