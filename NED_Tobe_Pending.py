from underthesea import ner
from wikipedia import *
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from uuid import uuid4
from tqdm import tqdm
from glob import glob
import pandas as pd 
import numpy as np
import torch
import faiss
import os
import re

wikipedia.set_lang("vi")

model_name = 'vinai/phobert-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def load_csv_data(dir="result"):
    """
    Load CSV files from a directory and combine them into a DataFrame.
    Each row gets a unique ID.
    """
    csv_files = glob(f"{dir}/*.csv")
    print(f"Found {len(csv_files)} CSV files in {dir}")

    all_dfs = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, encoding="utf-8-sig", sep=";")
        print(f" - {csv_file}: {len(df)} rows")
        all_dfs.append(df)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Combined dataset: {len(combined_df)} rows")

        combined_df["id"] = [str(uuid4()) for _ in range(len(combined_df))]

        return combined_df
    else:
        print("No data loaded.")
        return None

def load_abbreviation(path='lib/abbreviation.xlsx'):
    ab = pd.read_excel(path)
    abbreviations = dict(zip(ab['Abrreviation'], ab['Full']))
    return abbreviations 

def load_dictionary(path='lib/Dictionary_Underthesea.xlsx'):
    df = pd.read_excel(path)
    df.dropna(subset=["Entity", "Wikipedia Summary"], inplace=True)
    return df["Entity"].tolist(), df["Wikipedia Summary"].tolist()

def load_faiss_index(path='storage/faiss_index.index'):
    if os.path.exists(path):
        index = faiss.read_index(path)
        print(f"FAISS index loaded from {path}")
        return index
    else:
        raise FileNotFoundError(f"Index file not found at: {path}")
    
def load_summaries(path='storage/summaries.npy'):
    return np.load(path, allow_pickle=True).tolist()

def index_exists():
    return os.path.exists("storage/faiss_index.index") and os.path.exists("storage/summaries.npy")


def NER_and_map_sentences(text):
    # Join strings and clean the text
    # if (user_question): # if user prompt then join user prompt else join string from file
    #     text = user_question
    # else:
    #     text = " ".join(text['text'].astype(str))
    abbreviations = load_abbreviation()
    
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.strip() # Remove white spaces
    text = re.sub(r'[^\w\s.]', '', text)  # Remove special characters
    
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split by sentence-ending punctuation
    
    # Initialize result storage
    entity_sentence_map = []
    seen_pairs = set()

    # underthesea
    for sentence in tqdm(sentences, desc="Processing with NER from underthesea"):
        entities = ner(sentence)  # Perform NER on the sentence using underthesea
        entity_full= [entity[0] for entity in entities if entity[1] in ['N', 'Np']] # Select Noun or Speacial Name
        entity_full = [abbreviations.get(ent.lower().strip(), ent.strip()) for ent in entity_full] # Change entity with entity in abbreviation
        for entity in entity_full:
            pair = (entity,sentence)
            if pair not in seen_pairs:
                entity_sentence_map.append((entity, sentence))
                seen_pairs.add(pair)
    
    entity_sentence_map = [pair for pair in entity_sentence_map if len(pair[0]) > 1] # Remove Single character
    
    # Remove element that already exist in another element
    filter_list= []
    
    for pair in entity_sentence_map:
        is_unique = True
        for another_pair in entity_sentence_map:
            if pair[0]!=another_pair[0] and pair[0] in another_pair[0]:
                is_unique = False
                break
        if is_unique == True:
            filter_list.append(pair)
            
    entity_sentence_map=filter_list
    
    # Remove duplicates and sort results
    entity_sentence_map = list(set(entity_sentence_map))  # Remove duplicates
    entity_sentence_map.sort(key=lambda x: x[0].lower() if x[0] else '')  # Sort by entity name
    
    return entity_sentence_map
   
def wikipedia_result(Entity_sentence_map):
    Wiki_Result_List = []
    Unprocessed_Entities = []
    for pair in tqdm(Entity_sentence_map,desc="Processing Wikipidea"):
        try:
            summary = wikipedia.summary(pair[0], sentences=1)
            Wiki_Result_List.append((pair[0],pair[1], summary))

        except DisambiguationError as e:
            try:
                summary = wikipedia.summary(e.options[0], sentences=1)
                Wiki_Result_List.append((pair[0], pair[1], summary))
            except PageError:
                print(f"No page found for {pair[0]}")
                Unprocessed_Entities.append(pair)
                
    # print("Unprocessed Entities:", Unprocessed_Entities)
    return Wiki_Result_List

def encode_sentences(sentences, batch_size=32):
    all_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        all_embeddings.extend(embeddings)

    return np.array(all_embeddings)

def build_faiss_index(summaries=None):
    if index_exists():
        print("Loading FAISS index and summaries from cache...")
        index = load_faiss_index()
        summaries = load_summaries()
    else:
        print("No cached index found. Building FAISS index from dictionary...")
        entities, summaries = load_dictionary()
        print(f"Encoding {len(summaries)} summaries with batching...")
        summary_vectors = encode_sentences(summaries, batch_size=32)

        index = faiss.IndexFlatL2(summary_vectors.shape[1])
        index.add(np.ascontiguousarray(summary_vectors))
        
        faiss.write_index(index, 'storage/faiss_index.index')
        np.save('storage/summaries.npy', summaries)

        print("FAISS index built and saved.")

    return index

def search_summary_by_faiss(entity_sentence_map, summaries, summary_index):
    results = []

    for entity, sentence in tqdm(entity_sentence_map, desc="FAISS Lookup"):
        encoded = encode_sentences([entity])[0]
        D, I = summary_index.search(np.array([encoded]), 1)
        summary = summaries[I[0][0]]
        distance = D[0][0]
        results.append((entity, sentence, summary))
    
    return results


def Distance(Result_List):
    # Dictionary to store the shortest distance per entity
    entity_distances = {}

    # Iterate through each pair in Wiki_Result_List
    for entity, sentence, summary in tqdm(Result_List,desc="Calculating Distance"):
        # Encode sentences into vectors
        vector1 = encode_sentences([sentence])[0]  # Flatten to 1D
        vector2 = encode_sentences([summary])[0]  # Flatten to 1D
        
        # Compute cosine distance
        distance = cosine_distance(vector1, vector2)
        if distance < 0.5:
            # Store the entity with the shortest distance
            if entity not in entity_distances or distance < entity_distances[entity][3]:
                entity_distances[entity] = (entity, sentence, summary, distance)
    
    # Convert dictionary values to a list
    Distances = list(entity_distances.values())
    
    Result = [f"{item[0]} (Accuracy: {item[3]}): {item[2]} " for item in Distances]
    Result = "\n".join(Result)
    
    print (Result)   
    return Result
    
def cosine_distance(vector1, vector2):
    # Compute cosine similarity
    similarity = cosine_similarity([vector1], [vector2])
    return 1 - similarity[0][0]

def Distance_df(Distances):
    Distance_sentence_df = pd.DataFrame(Distances, columns=['Entity' ,'Sentence', 'Wikipedia Summary','Distance'])
    output_path = 'lib/Dictionary_Underthesea.xlsx'
    Distance_sentence_df.to_excel(output_path, index=False)
    print(f"Distances results exported to {output_path}")


def NED(text):
    
    entities, summaries = load_dictionary()
    Entity_sentence_map = NER_and_map_sentences(text)
    summary_index = build_faiss_index(summaries)
    
    # Result_List = wikipedia_result(Entity_sentence_map)
    Result_List = search_summary_by_faiss(Entity_sentence_map, summaries, summary_index)
    Result = Distance(Result_List)
    # Distance_df(Result)
    return Result
    
    

def main():
    # text = load_csv_data()
    text = input("Enter your questions: ")
    
    Result = NED(text)
    
     
    
if __name__ == "__main__":
    main()




