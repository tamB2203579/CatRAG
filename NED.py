from underthesea import ner
from wikipedia import *
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd 
import numpy as np
import faiss
import re

wikipedia.set_lang("vi")

# df = pd.read_csv('result/data.csv', encoding='utf-8')
# df = df.replace('\n', '', regex=True)

ab = pd.read_excel('lib/abbreviation.xlsx')
abbreviations = dict(zip(ab['Abrreviation'], ab['Full']))

model_name = 'vinai/phobert-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

user_question="học phí học phần đại cương là gì ?"

def NER():

    # text = " ".join(df['data'].astype(str)) # Join strings
    text = re.sub(r'\d+', '', text) # Remove number
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    entities = ner(text)


    Noun_Entity_List = [entity[0] for entity in entities if entity[1] in ['N', 'Np']] # Select Noun or Speacial Name
    Noun_Entity_List = list(set(Noun_Entity_List)) # Remove Identical Entity
    Noun_Entity_List = [entity for entity in Noun_Entity_List if len(entity) > 1] # Remove Single character

    # Change entity with entity in abbreviation
    for i, entity in enumerate(Noun_Entity_List):
        if entity in abbreviations:
            Noun_Entity_List[i] = abbreviations[entity]
    Noun_Entity_List = list(set(Noun_Entity_List)) # Remove duplicate element
            

    # Remove element that already exist in another element
    filter_list = []

    for entity in Noun_Entity_List:
        is_unique = True
        for another_entity in Noun_Entity_List:
            if entity!=another_entity and entity in another_entity:
                is_unique = False
                break
        if is_unique == True:
            filter_list.append(entity)
    Noun_Entity_List=filter_list

    Noun_Entity_List = sorted(Noun_Entity_List, key=str.lower) # Sort in alphabetical order
    
    return Noun_Entity_List

def NER_and_map_sentences(text):
    # Join strings and clean the text
    # if (user_question): # if user prompt then join user prompt else join string from file
    #     text = user_question
    # else:
    #     text = " ".join(df['data'].astype(str))
    
    
    # text = re.sub(r'\d+', '', text)  # Remove numbers
    # text = text.strip() # Remove white spaces
    # text = re.sub(r'[^\w\s.]', '', text)  # Remove special characters
    
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split by sentence-ending punctuation
    
    # Initialize result storage
    entity_sentence_map = []
    seen_pairs = set()

    for sentence in tqdm(sentences, desc="Processing NER"):
        entities = ner(sentence)  # Perform NER on the sentence
        entity_full= [entity[0] for entity in entities if entity[1] in ['N', 'Np']] # Select Noun or Speacial Name
        entity_full = [abbreviations.get(entity, entity) for entity in entity_full] # Change entity with entity in abbreviation
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
            except Exception as sub_e:
                print(f"Failed to resolve disambiguation for '{pair[0]}': {sub_e}")
                Unprocessed_Entities.append(pair[0])
        except PageError:
            print(f"No page found for {pair[0]}")
            Unprocessed_Entities.append(pair)
        except Exception as e:
            print(f"Error with entity {pair[0]}: {e}")
            Unprocessed_Entities.append(pair[0])
    print("Unprocessed Entities:", Unprocessed_Entities)
    return Wiki_Result_List

def encode_sentences(sentences):
    inputs = tokenizer(sentences, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)

    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return embeddings

def find_closest_sentence(input_sentence, db_vectors, database_sentences):
    input_vector = encode_sentences([input_sentence])
    # Using FAISS to search for the closet vector
    index = faiss.IndexFlatL2(db_vectors.shape[1])
    index.add(np.ascontiguousarray(db_vectors))
    D, I = index.search(input_vector, 1)
    closest_sentence = database_sentences[I[0][0]]
    return closest_sentence

def Distance(Wiki_Result_List):
    # Dictionary to store the shortest distance per entity
    entity_distances = {}

    # Iterate through each pair in Wiki_Result_List
    for entity, sentence, wiki_summary in tqdm(Wiki_Result_List,desc="Calculating Distance"):
        # Encode sentences into vectors
        vector1 = encode_sentences([sentence])[0]  # Flatten to 1D
        vector2 = encode_sentences([wiki_summary])[0]  # Flatten to 1D
        
        # Compute cosine distance
        distance = cosine_distance(vector1, vector2)
        if distance < 0.5:
            # Store the entity with the shortest distance
            if entity not in entity_distances or distance < entity_distances[entity][3]:
                entity_distances[entity] = (entity, sentence, wiki_summary, distance)
    
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



def NED(text):
    Entity_sentence_map = NER_and_map_sentences(text)
    Wiki_Result_List = wikipedia_result(Entity_sentence_map)
    Result = Distance(Wiki_Result_List)
    return Result
    
    

def main():
    
    NED("Thế nào là nghiên cứu khoa học và làm sao để kiếm giáo viên hướng dẫn ? Ngoài ra điều kiện để đạt học bổng học kì là gì ?")
    
    
    
if __name__ == "__main__":
    main()




