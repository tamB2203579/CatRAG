from llama_index.core import Document
from helper_functions import *

df = load_csv_data()

if df is not None:
    documents = [Document(text=row['text'], id_=row['id'], metadata={"label": row['label']}) for _,row in df.iterrows()]
    print(f"Created {len(documents)} documents for indexing")
    if documents:
        # create_vector_store(documents)
        loaded_index = load_vector_store()
    else:
        documents = []
        print("No documents created due to data loading issues")
    # populate_graph(df)
    # print("Knowledge graph construction completed")
else:
    print("Skipping knowledge graph construction due to missing data")

while True:
    query = input("Nhập câu hỏi của bạn: ")
    if query == "q":
        exit()
        break
    result = graphrag_chatbot(query)
    print(result["response"])