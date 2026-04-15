import json
from pathlib import Path

# Path to the downloaded SQuAD file
SQUAD_PATH = Path('data/train-v2.0.json')
# Output path for the converted subset
OUTPUT_PATH = Path('data/processed/squad_subset.json')

# Number of samples to extract
N_SAMPLES = 100

def convert_squad_to_custom_schema(squad_path, output_path, n_samples=100):
    with open(squad_path, 'r', encoding='utf-8') as f:
        squad = json.load(f)
    samples = []
    count = 0
    for article in squad['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                if qa.get('is_impossible', False):
                    continue  # skip unanswerable
                question = qa['question']
                answers = qa['answers']
                if not answers:
                    continue
                answer = answers[0]['text']
                sample = {
                    'query': question,
                    'answer': answer,
                    'subtasks': [
                        'Retrieve relevant documents',
                        'Reason over context',
                        'Generate final answer'
                    ],
                    'context_docs': [context],
                    'ground_truth': answer
                }
                samples.append(sample)
                count += 1
                if count >= n_samples:
                    break
            if count >= n_samples:
                break
        if count >= n_samples:
            break
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} samples to {output_path}")

if __name__ == "__main__":
    convert_squad_to_custom_schema(SQUAD_PATH, OUTPUT_PATH, N_SAMPLES)
