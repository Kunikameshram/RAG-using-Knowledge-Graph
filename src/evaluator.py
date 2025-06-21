import re
import pandas as pd
from rouge_score import rouge_scorer
from evaluate import load
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from .qa_chain import llm, ask_question

qag_prompt = PromptTemplate.from_template(
    """You are given a passage of text. Generate exactly 10 factual question-answer pairs based on it.

Passage:
{text}

Output format:
Q1: <question>
A1: <answer>
Q2: <question>
A2: <answer>
Q3: <question>
A3: <answer>"""
)

qag_chain = LLMChain(llm=llm, prompt=qag_prompt)

def generate_qag_pairs_from_text(text: str) -> str:
    return qag_chain.run({"text": text})

def parse_qa_pairs(text: str):
    pattern = re.findall(r"Q\d:\s*(.*?)\nA\d:\s*(.*?)(?=\nQ\d:|\Z)", text, re.DOTALL)
    return [{"question": q.strip(), "answer": a.strip()} for q, a in pattern]

rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
bertscore_metric = load("bertscore")

def evaluate_qa_pairs(true_answer: str, generated_answer: str):
    rouge_score = rouge.score(true_answer, generated_answer)
    rouge_l = rouge_score['rougeL'].fmeasure

    bert_result = bertscore_metric.compute(
        predictions=[generated_answer],
        references=[true_answer],
        lang="en"
    )
    bert_f1 = bert_result["f1"][0]

    return rouge_l, bert_f1

def evaluate_qa_on_text(text: str):
    generated_qas = generate_qag_pairs_from_text(text)
    qa_pairs = parse_qa_pairs(generated_qas)
    results = []

    for qa in qa_pairs:
        question = qa["question"]
        true_answer = qa["answer"]
        generated_answer = ask_question(question)
        rouge_l, bert_f1 = evaluate_qa_pairs(true_answer, generated_answer)
        results.append({
            "Question": question,
            "True Answer": true_answer,
            "Generated Answer": generated_answer,
            "ROUGE-L": round(rouge_l, 4),
            "BERTScore F1": round(bert_f1, 4)
        })
    return pd.DataFrame(results)
