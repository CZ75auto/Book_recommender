import pandas as pd
import gradio as gr
import numpy as np
import pickle
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

persist_directory = "./chroma_db"

raw_documents = TextLoader("tagged_descriptions.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

if os.path.exists(persist_directory) and os.listdir(persist_directory):
    # 如果存在並且有資料，就直接載入
    db_books = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
else:
    # 否則建立資料庫並持久化
    db_books = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory=persist_directory)



def retreive_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    inital_top_k: int = 50,
    final_top_k: int = 16,
    ) -> pd.DataFrame:

    print(db_books)

    recs = db_books.similarity_search(query, k=inital_top_k)

    books_list = [int(rec.page_content.strip('"').split(": ")[0].strip()) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)

    return book_recs

def recommend_books(
        query: str,
        category: str = None,
        tone: str = None,
        ):
    recommendations = retreive_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")

        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Sad", "Angry", "Suspenseful"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommendation System")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book : ", placeholder="e.g. A book about a forgiveness")

        category_dropdown = gr.Dropdown(label="Select a category : ", choices=categories, value="All")
        tone_dropdown = gr.Dropdown(label="Select a tone : ", choices=tones, value="All")

        submit_button = gr.Button("Get Recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended Books", columns = 8, rows = 2)

    submit_button.click(
        fn = recommend_books,
        inputs = [user_query, category_dropdown, tone_dropdown],
        outputs = output
    )

if __name__ == "__main__":
    dashboard.launch()