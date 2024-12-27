import json
from keybert import KeyBERT

class DatasetLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self):
        parsed_data = []
        with open(self.file_path, 'r') as f:
            for line in f:
                if line.strip():  # Ignore empty lines
                    data = json.loads(line)
                    # Extract only the required fields
                    parsed_data.append({key: data[key] for key in ["id", "title", "abstract"] if key in data})
        return parsed_data

def main():
    dataset_path = 'datasets/arxiv-metadata-oai-snapshot.json'
    try:
        dataset_loader = DatasetLoader(dataset_path)
    except FileNotFoundError:
        print("File not found.")
        return
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return

    # You can now use the loaded data to train your keyword extraction model
    kbert = KeyBERT()
    try:
        for item in dataset_loader.data:
            keywords = kbert.extract_keywords(item["abstract"], top_n=10)
            print(f"ID: {item['id']}, Keywords: {keywords}")
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    main()
