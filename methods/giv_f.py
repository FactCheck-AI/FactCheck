"""
GIV - Few shot (Guided Iterative Verification) Method
"""

import random
import time
from typing import Dict, List, Any, Optional

from data_loader import load_dataset
from llm_client import LLMClient
from methods.utils import save_results

# Define colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
END = "\033[0m"

dbp_examples = [
    {
        'fact': {
            's': 'Monica Bellucci',
            'p': {
                'DBpedia': 'birthPlace',
                'FactBench': 'birthPlace',
                'YAGO': 'wasBornIn'
            },
            'o': 'Città di Castello, Italy'
        },
        'response': {
            'agreement': 'A',
            'correctness': 'T'
        }
    },
    {
        'fact': {
            's': 'Curiosity (rover)',
            'p': {
                'DBpedia': 'location',
                'FactBench': 'location',
                'YAGO': 'isLocatedIn'
            },
            'o': 'Mars'
        },
        'response': {
            'agreement': 'A',
            'correctness': 'T'
        }
    },
    {
        'fact': {
            's': 'Isaac Newton',
            'p': {
                'DBpedia': 'knownFor',
                'FactBench': 'knownFor',
                'YAGO': 'isKnownFor'
            },
            'o': 'Universal gravitation'
        },
        'response': {
            'agreement': 'A',
            'correctness': 'T'
        }
    },
    {
        'fact': {
            's': 'La Rambla',
            'p': {
                'DBpedia': 'locationCity',
                'FactBench': 'location',
                'YAGO': 'isLocatedIn'
            },
            'o': 'Berlin'
        },
        'response': {
            'agreement': 'D',
            'correctness': 'F'
        }
    },
    {
        'fact': {
            's': 'J. K. Rowling',
            'p': {
                'DBpedia': 'author',
                'FactBench': 'author',
                'YAGO': 'author'
            },
            'o': 'Lord of The Rings'
        },
        'response': {
            'agreement': 'D',
            'correctness': 'F'
        }
    },
    {
        'fact': {
            's': 'Hyundai Motor Company',
            'p': {
                'DBpedia': 'foundationPlace',
                'FactBench': 'foundationPlace',
                'YAGO': 'locationCreated'
            },
            'o': 'Rio de Janeiro'
        },
        'response': {
            'agreement': 'D',
            'correctness': 'F'
        }
    }
]


def create_examples(examples, dataset, task='correctness'):
    random.shuffle(examples)
    fact = """### Fact
{s} {p} {o}
Judgment:"""
    return "".join(
        fact.format(
            s=example['fact']['s'],
            p=(
                example['fact']['p']
                if dataset == 'NELL'
                else example['fact']['p'][dataset]
            ),
            # NELL doesn't have the inner dict on the predicate
            o=example['fact']['o'],
        )
        + ' '
        + example['response'][task]
        + '\n'
        for example in examples
    )


def create_giv_f_prompt(fact: Dict[str, Any], dataset_name: str) -> str:
    """
    Create GIV_F-specific prompt for fact verification

    GIV relies on direct knowledge assessment without external resources -- examples are provided
    """
    s = fact.get("s", "")
    p = fact.get("p", "")
    o = fact.get("o", "")

    labels = '"T", "F"'
    responses = '"T" (true), or "F" (false)'

    examples = create_examples(dbp_examples, dataset_name)
    return """You are an expert annotator for the fact verification task.
    You will be provided a fact in the form of a subject, predicate, and object.
    Your job is to assess the fact based solely on the knowledge you have been trained on, without using any external information.

    ### Guidelines
    {guidelines}

    Please do not spread incorrect information. Your judgment must rely only on your internal knowledge.
    You must always answer in English.
    The answer must contain only the judgment ({judgments}). Lastly, keep your answer concise.

    ### Retry Mechanism
    If your response does not comply with the guidelines above, you will be asked to retry.
    In the retry follow-up, you must meet the same guidelines and provide a compliant answer.
    You will have a total of {max_attempts} retry attempts.
    If you fail to provide a compliant answer within the attempts, your final response for the fact will be set to "NA" (Not an Answer).

    ### Examples
    {examples}

    ### Input format
    The fact is given in one line: first the subject, then the predicate and finally the object.

    ### Fact
    {s} {p} {o}

    Judgment:

    Your previous response did not meet the guidelines, please try again and provide a compliant answer.
    You have {attempts} remaining attempts.
    If you fail to provide a compliant answer within the remaining attempts, your final response for the fact will be set to "NA" (Not an Answer).

    ### Reminder of the Guidelines
    The answer should be {responses} based on your internal knowledge. Limit the use of "N" as much as possible.
    Provide your judgment in English and keep it concise. Do not spread incorrect information.
    """.format(
        max_attempts=3,
        judgments=labels,
        attempts=3,
        examples=examples,
        responses=responses,
        s=s,
        p=p,
        o=o,
        guidelines=(
            'The answer should be "T" (true) if the fact is accurate based on your knowledge up to 2015, or "F" (false) if the fact is inaccurate based on your knowledge up to 2015.'
            if dataset_name == 'DBpedia'
            else 'The answer should be "T" (true) if the fact is accurate based on your knowledge, or "F" (false) if the fact is inaccurate based on your knowledge.'
        ),
    )


def run_giv_f_method(config: Dict[str, Any]) -> None:
    """
    Run GIV_F (Direct Knowledge Assessment) method

    GIV directly queries the LLM's internal knowledge without external retrieval
    """
    print(f"\n{BOLD}{BLUE}🧠 Running GIV (Direct Knowledge Assessment) Method{END}")

    # Extract configuration
    dataset_name = config.get("dataset", {}).get("name", "Unknown")
    kg_ids = config.get("knowledge_graph", {}).get("kg_ids")
    llm_config = config.get("llm", {})

    print(f"{CYAN}Method: Direct Knowledge Assessment{END}")
    print(f"{CYAN}Dataset: {dataset_name}{END}")
    print(f"{CYAN}LLM: {llm_config.get('model', 'Unknown')} ({llm_config.get('mode', 'Unknown')}){END}")

    # Load dataset
    print(f"\n{BOLD}📚 Loading Dataset...{END}")
    facts = load_dataset(dataset_name, kg_ids=kg_ids)

    if not facts:
        print(f"{RED}✗ No facts to process. Exiting GIV method.{END}")
        return

    # Initialize LLM client
    print(f"\n{BOLD}🤖 Initializing LLM Client...{END}")
    try:
        llm_client = LLMClient(config)

        print(f"{GREEN}✓ LLM client initialized for GIV method{END}")

    except Exception as e:
        print(f"{RED}✗ Failed to initialize LLM client: {str(e)}{END}")
        return

    # Process facts using GIV approach
    print(f"\n{BOLD}🔍 Processing Facts with GIV Method...{END}")
    print(f"{CYAN}GIV relies on the model's internal knowledge without external retrieval{END}")

    results = []
    total_cost = 0.0
    total_tokens = 0
    successful_predictions = 0

    start_time = time.time()

    for i, fact_kg in enumerate(facts):
        identifier, fact = fact_kg
        print(f"\n{BLUE}Processing fact {i+1}/{len(facts)}{END}")
        print(f"  Fact: {fact.get('s', '')} {fact.get('p', '')} {fact.get('o', '')}")

        # Create GIV_F-specific prompt
        prompt = create_giv_f_prompt(fact, dataset_name)

        # Get LLM response
        response = llm_client.generate_response(prompt)

        # Process result
        result = {
            "id": identifier,
            "method": "GIV-F",
            "fact": {
                "s": fact.get("s", ""),
                "p": fact.get("p", ""),
                "o": fact.get("o", ""),
                "label": "T" if fact.get("label") else "F"  # Ground truth if available
            },
            "prompt": prompt,
            "response": response.content,
            "success": response.success,
            "error_message": response.error_message,
            "response_time": response.response_time,
            # "tokens_used": response.tokens_used,
            # "cost": response.cost or 0.0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        results.append(result)

        # Update totals
        # if response.tokens_used:
        #     total_tokens += response.tokens_used
        # if response.cost:
        #     total_cost += response.cost
        if response.success:
            successful_predictions += 1

        # Print result
        if response.success:
            predicted = response.content.strip().upper()
            print(f"  {GREEN}GIV Prediction: {predicted}{END}")

            # Show accuracy if ground truth is available
            if "label" in fact:
                actual = "T" if fact["label"] else "F"
                correct = "✓" if predicted == actual else "✗"
                color = GREEN if predicted == actual else RED
                print(f"  {color}Ground Truth: {actual} {correct}{END}")
        else:
            print(f"  {RED}Failed: {response.error_message}{END}")

    # Calculate final metrics
    end_time = time.time()
    duration = end_time - start_time
    # accuracy = calculate_accuracy(results)

    # Print GIB summary
    print(f"\n{BOLD}{GREEN}📊 GIB Method Summary{END}")
    print(f"  Method: Direct Knowledge Assessment")
    print(f"  Total facts processed: {len(facts)}")
    print(f"  Successful predictions: {successful_predictions}")
    print(f"  Failed predictions: {len(facts) - successful_predictions}")
    # if accuracy is not None:
    #     print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Total tokens used: {total_tokens}")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Avg time per fact: {duration/len(facts):.2f} seconds")

    # Save results
    print(f"\n{BOLD}💾 Saving GIV_F Results...{END}")
    results_file = save_results('giv-f', results, config)

    if results_file:
        print(f"{GREEN}✅ GIV_F method completed successfully!{END}")
    else:
        print(f"{YELLOW}⚠️ GIV_F method completed but failed to save results{END}")