import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from llama_parse import LlamaParse
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from datasets import Dataset, load_dataset
from giskard.rag import KnowledgeBase, generate_testset, evaluate as giskard_evaluate
from ragas import evaluate as ragas_evaluate 
from ragas.metrics import (faithfulness, answer_relevancy, context_precision, context_recall)
from dotenv import load_dotenv
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

# Load environment variables
load_dotenv()

def parse_pdf_to_markdown(pdf_filepath: str, output_dir: str) -> str:
    if not os.path.exists(pdf_filepath):
        print(f"The file {pdf_filepath} does not exist. Please provide a valid path.")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    markdown_output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pdf_filepath))[0]}.md")

    if os.path.exists(markdown_output_path):
        print(f"Markdown file already exists at: {markdown_output_path}. Using the existing file.")
    else:
        print("Parsing PDF and generating Markdown file...")
        parser = LlamaParse(result_type="markdown", num_workers=4, verbose=True, language="en")
        documents = parser.load_data(pdf_filepath)

        with open(markdown_output_path, 'w', encoding='utf-8') as file:
            for doc in documents:
                if doc.text.strip():
                    file.write(doc.text + "\n\n")
        print(f"Markdown file saved at: {markdown_output_path}")

    return markdown_output_path

def split_markdown_to_chunks(md_content: str) -> list:
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    return markdown_splitter.split_text(md_content)

def create_vector_store(md_header_chunks: list) -> FAISS:
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(md_header_chunks, embeddings)
    return vectorstore.as_retriever()

def create_knowledge_base(md_header_chunks) -> KnowledgeBase:
    df = pd.DataFrame([doc.page_content for doc in md_header_chunks], columns=["text"])
    return KnowledgeBase(df)

def generate_testset_from_knowledge_base(knowledge_base: KnowledgeBase) -> Dataset:
    testset = generate_testset(
        knowledge_base,
        num_questions=2,
        agent_description="A chatbot answering questions about the context available"
    )
    testset.save("testset.jsonl")
    return testset

def create_rag_chain(retriever) -> callable:
    template = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use ten sentences maximum and keep the answer as per the retrieved context.
    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm_model = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4o-mini")
    output_parser = StrOutputParser()
    
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm_model
        | output_parser
    )

def evaluate_giskard(answer_fn, testset, knowledge_base):
    report = giskard_evaluate(answer_fn, testset=testset, knowledge_base=knowledge_base)
    giskard_df = report.to_pandas()
    giskard_df.to_csv("giskard_response.csv", index=False)

    # Generate Giskard evaluation heatmap
    correctness = report.correctness_by_question_type().values.reshape(-1, 1)
    plot_heatmap(correctness, "Giskard Correctness", "giskard_heatmap.pdf")

def evaluate_ragas(dataset):
    ragas_report = ragas_evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    ragas_df = ragas_report.to_pandas()
    ragas_df.to_csv("ragas_result.csv", index=False)

    # Generate RAGAS evaluation heatmap
    metrics = ragas_df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].values
    plot_heatmap(metrics, "RAGAS Metrics", "ragas_heatmap.pdf")

def plot_heatmap(data, title, output_path):
    os.makedirs("results", exist_ok=True)
    pdf_path = os.path.join("results", output_path)
    
    cmap = LinearSegmentedColormap.from_list("red_green", ["red", "green"])
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5)
    plt.title(title)
    
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(bbox_inches='tight')
    plt.close()

def main():
    pdf_filepath = input("Enter the path to the PDF file: ")
    local_dir = os.path.expanduser("~/Documents/pdf_to_md_files")
    markdown_output_path = parse_pdf_to_markdown(pdf_filepath, local_dir)
    if markdown_output_path is None:
        return

    with open(markdown_output_path, 'r', encoding='utf-8') as file:
        md_document_content = file.read()

    md_header_chunks = split_markdown_to_chunks(md_document_content)
    retriever = create_vector_store(md_header_chunks)
    knowledge_base = create_knowledge_base(md_header_chunks)
    testset = generate_testset_from_knowledge_base(knowledge_base)

    rag_chain = create_rag_chain(retriever)
    test_df = testset.to_pandas()
    test_questions = test_df["question"].values.tolist()
    test_groundtruths = test_df["reference_context"].values.tolist()
    
    data = {
        "question": test_questions,
        "answer": [rag_chain.invoke(query) for query in test_questions],
        "contexts": [[doc.page_content for doc in retriever.get_relevant_documents(query)] for query in test_questions],
        "ground_truth": test_groundtruths,
    }
    dataset = Dataset.from_dict(data)
    dataset.to_pandas().to_csv("response.csv", index=False)
    print("Response saved as CSV at: response.csv")

    answer_fn = lambda question, history=None: rag_chain.invoke(question)
    evaluate_giskard(answer_fn, testset, knowledge_base)
    evaluate_ragas(dataset)

if __name__ == "__main__":
    main()
