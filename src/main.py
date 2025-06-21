from src.config import *
from src.data_loader import load_and_split_documents
from src.graph_builder import build_and_add_graph_documents
from src.evaluator import evaluate_qa_on_text

def main():
    documents = load_and_split_documents("List of presidents of the United States")
    build_and_add_graph_documents(documents)
    df = evaluate_qa_on_text(documents[0].page_content)
    print(df)
    print(f"Average ROUGE-L Score: {df['ROUGE-L'].mean():.2f}")
    print(f"Average BERTScore F1: {df['BERTScore F1'].mean():.4f}")

if __name__ == "__main__":
    main()
