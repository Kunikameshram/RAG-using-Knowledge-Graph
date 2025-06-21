from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from typing import List

def load_and_split_documents(query: str, chunk_size=512, chunk_overlap=24) -> List:
    raw_documents = WikipediaLoader(query=query).load()
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(raw_documents)