import json

import json_repair

from data_loader import load_dataset

MODELS = ['gemma2-9b-it', 'llama3.1-8b-it', 'mistral-7b-it', 'qwen2.5-7b-it']
DATASETS = ['DBpedia', 'YAGO', 'FactBench']
for model in MODELS:
    model_name = model.replace("-it", "").replace("-", ":")
    for dataset in DATASETS:
        with open(f'results/{model}/iterative/correctness/{dataset}/fewshot_natural_inline.json', 'r') as f:
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
            # extracted_ans = get_answer_from_llm(answer['full_ans'], gt[identifier])
            # if answer['short_ans'] == -1:
            #     if gt[identifier] == "F":
            #         actual_ans = "T"
            #     else:
            #         actual_ans = "F"
            # else:
            #     actual_ans = "T" if answer['short_ans'] else "F"
            #
            # if extracted_ans != actual_ans:
            #     print(f"Mismatch for {identifier}: Extracted: {extracted_ans}, Actual: {actual_ans}")
            #     # print(f"Full Answer: {answer['full_ans']}")
            #
            corrected_answers.append({
                    "id": identifier,
                    "method": "DKA",
                    # "fact": {
                    #     "s": "Roscoe Arbuckle",
                    #     "p": "deathPlace",
                    #     "o": "New York City",
                    #     "label": 1
                    # },
                    "response_time": answer['time'],
                    "response": "T" if answer['label'] == 'true' else "F",
                    "success": True,
                    "timestamp": "2025-05-27 10:37:10"
                }
            )

        with open(f'results/paper_results/GIV-F/{dataset}_open-source_{model_name}_giv-f.json', 'w') as f:
            json.dump(corrected_answers, f, indent=4)