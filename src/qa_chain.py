from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from typing import List, Tuple
from .retriever import retriever, format_chat_history

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

CONDENSE_QUESTION_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)

_search_query = RunnableBranch(
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
        RunnablePassthrough.assign(chat_history=lambda x: format_chat_history(x["chat_history"]))
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    ),
    RunnableLambda(lambda x: x["question"]),
)

ANSWER_PROMPT_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_PROMPT_TEMPLATE)

chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | ANSWER_PROMPT
    | llm
    | StrOutputParser()
)

def ask_question(question: str, chat_history: List[Tuple[str, str]] = None) -> str:
    inputs = {"question": question}
    if chat_history:
        inputs["chat_history"] = chat_history
    return chain.invoke(inputs)
