import json

import json_repair

from data_loader import load_dataset

model_name = 'llama3.1'
dataset = 'DBpedia'
with open(f'results/paper_results/{dataset}_{model_name}_best_model.json', 'r') as f:
    data = json.load(f)

corrected_answers = []

def get_answer_from_llm(json_string: str, actual_answer: str) -> str:
    decoded_object = json_repair.repair_json(json_string, return_objects=True)
    # print(f"Decoded Object: {decoded_object}")
    # if decoded_object is [], loop over it
    if isinstance(decoded_object, list):
        for obj in decoded_object:
            if not 'output' in str(obj):
                continue
            try:
                if obj['output'] == 'yes':
                    return "T"
                elif obj['output'] == 'no':
                    return "F"
            except Exception:
                continue

    elif not 'output' in decoded_object:
        pass
    elif decoded_object['output'] == 'yes':
        return "T"
    elif decoded_object['output'] == 'no':
        return "F"

    if actual_answer == 'T':
        return "F"

    return "T"

gt = {}
kg = load_dataset(dataset)
for identifier, kg in kg:
    gt[identifier] = "T" if kg['label'] else "F"

for identifier, answer in data.items():

    extracted_ans = get_answer_from_llm(answer['full_ans'], gt[identifier])
    if answer['short_ans'] == -1:
        if gt[identifier] == "F":
            actual_ans = "T"
        else:
            actual_ans = "F"
    else:
        actual_ans = "T" if answer['short_ans'] else "F"

    if extracted_ans != actual_ans:
        print(f"Mismatch for {identifier}: Extracted: {extracted_ans}, Actual: {actual_ans}")
        # print(f"Full Answer: {answer['full_ans']}")

    corrected_answers.append({
            "id": identifier,
            "method": "RAG",
            # "fact": {
            #     "s": "Roscoe Arbuckle",
            #     "p": "deathPlace",
            #     "o": "New York City",
            #     "label": 1
            # },
            "full_answer": answer['full_ans'],
            "response": actual_ans,
            "success": True,
            "timestamp": "2025-05-27 10:37:10"
        }
    )

with open(f'results/paper_results/{dataset}_{model_name}_corrected_answers.json', 'w') as f:
    json.dump(corrected_answers, f, indent=4)