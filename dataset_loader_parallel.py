import json
from keybert import KeyBERT
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import signal
import sys

class DatasetLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        with open(self.file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

def process_line(line):
    try:
        data = json.loads(line)
        if all(key in data for key in ["id", "abstract"]):
            kbert = KeyBERT()
            keywords = kbert.extract_keywords(data["abstract"], top_n=10)
            return {"id": data["id"], "keywords": keywords}
    except json.JSONDecodeError:
        return None  # Skip invalid JSON lines
    return None

def handle_interrupt(signal, frame):
    print("\nExecution interrupted, shutting down...")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, handle_interrupt)  # Graceful exit on Ctrl+C

    dataset_path = 'datasets/arxiv-metadata-oai-snapshot.json'
    try:
        dataset_loader = DatasetLoader(dataset_path)
        lines = dataset_loader.load_data()
    except FileNotFoundError:
        print("File not found.")
        return

    results = []
    with ProcessPoolExecutor() as executor:
        try:
            for result in executor.map(process_line, lines):
                if result:
                    results.append(result)
        except KeyboardInterrupt:
            print("Processing was interrupted.")
            executor.shutdown(wait=False)  # Stop workers immediately

    # Output the results
    for res in results:
        print(f"ID: {res['id']}, Keywords: {res['keywords']}")

if __name__ == "__main__":
    main()
