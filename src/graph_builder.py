from src.config import OPENAI_API_KEY
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
llm_transformer = LLMGraphTransformer(llm=llm)
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database="neo4j"
)

def build_and_add_graph_documents(documents):
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )