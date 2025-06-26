from uuid import uuid4
import datetime
import json
import os

class HistoryManager:
    def __init__(self, history_dir="history"):
        self.history_dir = history_dir
        self.history = {}
        self.max_entries = 10
        
        # Create history directory if it doesn't exist
        os.makedirs(history_dir, exist_ok=True)
        
    def generate_session_id(self):
        """Generate a unique session ID"""
        return str(uuid4())
        
    def add_to_history(self, session_id, query, response):
        """Add a query-response pair to the session history"""
        if session_id not in self.history:
            self.history[session_id] = []
        
        entry = {
            "query": query,
            "response": response,
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        if len(self.history[session_id]) >= self.max_entries:
            self.history[session_id].pop(0)
        self.history[session_id].append(entry)
        self.save_history_to_file(session_id)
        
    def save_history_to_file(self, session_id):
        """Save the session history to a JSON file"""
        filename = f"{self.history_dir}/{session_id}.json"

        # Load the existing full history from file (if any)
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                full_history = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            full_history = []

        new_entries = [
            entry for entry in self.history.get(session_id, [])
            if (entry['query'], entry['timestamp']) 
        ]

        full_history.extend(new_entries)

        try:
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(full_history, file, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save history to {filename}: {e}")
            
    def load_history_from_file(self, session_id):
        """Load the session history from a JSON file"""
        filename = f"{self.history_dir}/{session_id}.json"
        
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                loaded_history = list(json.load(file))
                self.history[session_id] = loaded_history[-self.max_entries:]
                
            print(f"History loaded from {filename} successfully!")
            
        except FileNotFoundError:
            print(f"No history file found at {filename}. Starting with empty history.")
            self.history[session_id] = []
        except Exception as e:
            print(f"Failed to load history from {filename}: {e}")
            self.history[session_id] = []
            
    def clear_history(self, session_id):
        """Clear the session history"""
        filename = f"{self.history_dir}/{session_id}.json"
        if session_id in self.history:
            self.history[session_id].clear()
            
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump([], file, indent=4)
            print(f"History and file {filename} have been cleared successfully!")
            
    def get_history(self, session_id):
        """Get the session history"""
        return self.history.get(session_id, [])
