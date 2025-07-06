from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import List, Dict
from glob import glob
import pandas as pd
import random
import json
import os

load_dotenv()

class QuestionGenerator:
    def __init__(self, model_name="gpt-4o-mini"):
        """Initialize the question generator with OpenAI model"""
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.prompt_template = self._load_prompt_template()
        
    def _load_prompt_template(self):
        """Load the question generation prompt template"""
        with open("prompt/question_generate.txt", "r", encoding="utf-8") as f:
            template = f.read()
        return ChatPromptTemplate.from_template(template)
    
    def read_csv_files(self, result_folder="result"):
        """Read all CSV files from the result folder and combine them"""
        csv_files = glob(f"{result_folder}/*.csv")
        print(f"Found {len(csv_files)} CSV files in {result_folder}")
        
        all_chunks = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, encoding="utf-8-sig", sep=";")
                print(f"- {csv_file}: {len(df)} rows")
                
                # Convert DataFrame to list of dictionaries
                for _, row in df.iterrows():
                    chunk_data = {
                        "text": row.get("text", ""),
                        "source_file": os.path.basename(csv_file)
                    }
                    all_chunks.append(chunk_data)
                    
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue
        
        print(f"Total chunks loaded: {len(all_chunks)}")
        return all_chunks
    
    def generate_questions_for_chunk(self, chunk_text: str) -> List[Dict]:
        """Generate questions and answers for a single chunk of text"""
        try:
            chain = (
                self.prompt_template
                | self.llm
                | StrOutputParser()
            )
            
            response = chain.invoke({"context": chunk_text})
            
            # Parse the JSON response
            qa_pairs = []
            try:
                # Clean up the response to extract JSON
                response_clean = response.strip()
                
                # Find the JSON array in the response
                start_idx = response_clean.find('[')
                end_idx = response_clean.rfind(']') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response_clean[start_idx:end_idx]
                    parsed_data = json.loads(json_str)
                    
                    # Validate and clean the parsed data
                    for item in parsed_data:
                        if isinstance(item, dict) and 'question' in item and 'answer' in item:
                            question = item['question'].strip()
                            answer = item['answer'].strip()
                            
                            # Filter out empty or very short questions/answers
                            if question and answer and len(question) > 10 and len(answer) > 5:
                                qa_pairs.append({
                                    'question': question,
                                    'answer': answer
                                })
                else:
                    print(f"Could not find JSON array in response: {response[:100]}...")
                    
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Response: {response[:200]}...")
                
            return qa_pairs
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            return []
    
    def save_to_excel(self, evaluation_data, output_file="dataset.xlsx"):
        """Save evaluation data to Excel file with question and ground_truth columns"""
        print(f"Saving data to Excel file: {output_file}")
        
        # Prepare data for Excel
        excel_data = []
        
        # Handle different data formats
        if isinstance(evaluation_data, dict):
            # If data is organized by label
            for questions in evaluation_data.items():
                for item in questions:
                    excel_data.append({
                        'question': item['question'],
                        'ground_truth': item['answer']
                    })
        elif isinstance(evaluation_data, list):
            # If data is a simple list
            for item in evaluation_data:
                excel_data.append({
                    'question': item['question'],
                    'ground_truth': item['answer']
                })
        
        # Create DataFrame and save to Excel
        df = pd.DataFrame(excel_data)
        df.to_excel(output_file, index=False, engine='openpyxl')
        
        print(f"Successfully saved {len(excel_data)} questions to {output_file}")
        return df
    
    def generate_evaluation_dataset(self, num_samples=50, output_file="evaluation_questions.json", save_excel=False, excel_file="dataset.xlsx"):
        """Generate evaluation questions from random chunks"""
        print("Starting evaluation dataset generation...")
        
        # Read all chunks
        chunks = self.read_csv_files()
        
        if not chunks:
            print("No chunks found. Exiting.")
            return
        
        # Randomly sample chunks
        sample_chunks = random.sample(chunks, min(num_samples, len(chunks)))
        
        evaluation_data = []
        
        for i, chunk in enumerate(sample_chunks):
            print(f"Processing chunk {i+1}/{len(sample_chunks)}...")
            
            qa_pairs = self.generate_questions_for_chunk(chunk["text"])
            
            if qa_pairs:
                for qa in qa_pairs:
                    eval_item = {
                        "id": f"eval_{i+1}_{len(evaluation_data)+1}",
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "context": chunk["text"],
                        "source_file": chunk["source_file"]
                    }
                    evaluation_data.append(eval_item)
        
        # Save to JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
        
        print(f"Generated {len(evaluation_data)} evaluation questions")
        print(f"Saved to: {output_file}")
        
        # Save to Excel if requested
        if save_excel:
            self.save_to_excel(evaluation_data, excel_file)
        
        return evaluation_data
    
    def generate_dataset_excel_only(self, num_samples=50, excel_file="dataset.xlsx"):
        """Generate evaluation dataset and save only to Excel file"""
        print("Generating evaluation dataset for Excel output...")
        
        # Read all chunks
        chunks = self.read_csv_files()
        
        if not chunks:
            print("No chunks found. Exiting.")
            return
        
        # Randomly sample chunks
        sample_chunks = random.sample(chunks, min(num_samples, len(chunks)))
        
        evaluation_data = []
        
        for i, chunk in enumerate(sample_chunks):
            print(f"Processing chunk {i+1}/{len(sample_chunks)}...")
            
            qa_pairs = self.generate_questions_for_chunk(chunk["text"])
            
            if qa_pairs:
                for qa in qa_pairs:
                    eval_item = {
                        "question": qa["question"],
                        "answer": qa["answer"]
                    }
                    evaluation_data.append(eval_item)
        
        # Save directly to Excel
        self.save_to_excel(evaluation_data, excel_file)
        
        return evaluation_data

    def preview_questions(self, num_samples=5):
        """Preview generated questions from a few random chunks"""
        print("Generating preview questions...")
        
        chunks = self.read_csv_files()
        
        if not chunks:
            print("No chunks found.")
            return
        
        sample_chunks = random.sample(chunks, min(num_samples, len(chunks)))
        
        for i, chunk in enumerate(sample_chunks):
            print(f"\n{'='*50}")
            print(f"Source: {chunk['source_file']}")
            print(f"{'='*50}")
            print(f"Text: {chunk['text'][:200]}...")
            print(f"\nGenerated Questions and Answers:")
            
            qa_pairs = self.generate_questions_for_chunk(chunk["text"])
            
            if qa_pairs:
                for j, qa in enumerate(qa_pairs, 1):
                    print(f"{j}. Q: {qa['question']}")
                    print(f"   A: {qa['answer']}")
                    print()
            else:
                print("No questions generated for this chunk.")


def main():
    """Main function to run the question generation"""
    generator = QuestionGenerator()
    
    print("Question Generation for RAG Evaluation")
    print("="*50)
    
    choice = input("""
        Choose an option:
        1. Preview questions from random chunks
        2. Generate evaluation dataset (random sampling)
        3. Generate dataset and save to Excel (question & ground_truth columns)
        4. Exit

        Enter your choice (1-4): """)
    
    if choice == "1":
        num_samples = int(input("Number of chunks to preview (default 5): ") or "5")
        generator.preview_questions(num_samples)
        
    elif choice == "2":
        num_samples = int(input("Number of chunks to process (default 50): ") or "50")
        output_file = input("Output file name (default: evaluation_questions.json): ") or "evaluation_questions.json"
        save_excel = input("Also save to Excel? (y/n, default: n): ").lower() == 'y'
        excel_file = "dataset.xlsx"
        if save_excel:
            excel_file = input("Excel file name (default: dataset.xlsx): ") or "dataset.xlsx"
        generator.generate_evaluation_dataset(num_samples, output_file, save_excel, excel_file)
        
    elif choice == "3":
        num_samples = int(input("Number of chunks to process (default 50): ") or "50")
        excel_file = input("Excel file name (default: dataset.xlsx): ") or "dataset.xlsx"
        generator.generate_dataset_excel_only(num_samples, excel_file)
        
    elif choice == "4":
        print("Goodbye!")
        return
        
    else:
        print("Invalid choice. Please try again.")
        main()


if __name__ == "__main__":
    main()
