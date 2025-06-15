import json
import pandas as pd
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv



load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")
df = pd.read_excel("result/dataset.xlsx", sheet_name="Khac")

with open("prompt/augment.txt", "r", encoding="utf-8") as f:
    template = f.read()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
prompt = ChatPromptTemplate.from_template(template)

chain = (
    prompt
    | llm
    | StrOutputParser()
)

augmented_data = []

category = df['label'].iloc[0]

num_variations = 9

for index, row in tqdm(df.iterrows(), total=len(df)):
    text = row['text']
    
    chain_input = {
        "category": category,
        "num_variations": num_variations,
        "text": text
    }
    
    try:
        result = chain.invoke(chain_input)
        
        augmented_texts = json.loads(result)
        
        augmented_data.append({
            "text": text,
            "label": row['label']
        })
        
        for variant in augmented_texts[0]["variants"]:
            augmented_data.append({
                "text": variant,
                "label": row['label']
            })
        
    except Exception as e:
        print(f"Error processing text at index {index}: {e}")

augmented_df = pd.DataFrame(augmented_data)
augmented_df.to_excel("augmented_dataset.xlsx", index=False)
print(f"Original dataset size: {len(df)}, Augmented dataset size: {len(augmented_df)}")