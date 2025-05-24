"""
DKA (Direct Knowledge Assessment) Method Implementation
"""

import json
import time
import os
from typing import Dict, List, Any, Optional

from data_loader import load_dataset
from llm_client import LLMClient

# Define colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
END = "\033[0m"


def save_dka_results(results: List[Dict[str, Any]], config: Dict[str, Any]) -> Optional[str]:
    """
    Save DKA results to a file

    Args:
        results (List[Dict[str, Any]]): List of results from the DKA method
        config (Dict[str, Any]): Configuration dictionary

    Returns:
        Optional[str]: Path to the saved results file
    """
    output_dir = config.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)
    dataset = config.get("dataset", {}).get("name")
    llm_config = config.get("llm", {})
    llm_model = llm_config.get("model")
    llm_mode = llm_config.get("mode").replace("_", "-")

    results_file = os.path.join(
        output_dir,
        f'{dataset}_{llm_mode}_{llm_model}_dka_{time.strftime("%Y%m%d-%H%M%S")}.json'
    )

    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"{GREEN}✓ DKA results saved to {results_file}{END}")
        return results_file
    except Exception as e:
        print(f"{RED}✗ Failed to save DKA results: {str(e)}{END}")
        return None

def create_dka_prompt(fact: Dict[str, Any], dataset_name: str) -> str:
    """
    Create DKA-specific prompt for fact verification
    
    DKA relies on direct knowledge assessment without external resources
    """
    s = fact.get("s", "")
    p = fact.get("p", "")
    o = fact.get("o", "")

    if dataset_name.lower().startswith('dbpedia'):
        prompt = f"""Your knowledge is limited to the year 2015. Only reply with "T" or "F" and nothing else.
If the user provides a fact you judge accurate (based on your knowledge up to 2015), reply with the
letter "T".
If the user provides a fact you judge inaccurate (based on your knowledge up to 2015), reply with the
letter "F".

Fact: {s} {p} {o}

Judgment:
"""
    else:
        prompt = f"""Only reply with "T" or "F" and nothing else.
If the user provides a fact you judge accurate, reply with the letter "T".
If the user provides a fact you judge inaccurate, reply with the letter "F".

Fact: {s} {p} {o}

Judgment:
"""

    return prompt


def run_dka_method(config: Dict[str, Any]) -> None:
    """
    Run DKA (Direct Knowledge Assessment) method
    
    DKA directly queries the LLM's internal knowledge without external retrieval
    """
    print(f"\n{BOLD}{BLUE}🧠 Running DKA (Direct Knowledge Assessment) Method{END}")

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
        print(f"{RED}✗ No facts to process. Exiting DKA method.{END}")
        return

    # Initialize LLM client
    print(f"\n{BOLD}🤖 Initializing LLM Client...{END}")
    try:
        llm_client = LLMClient(config)

        print(f"{GREEN}✓ LLM client initialized for DKA method{END}")

    except Exception as e:
        print(f"{RED}✗ Failed to initialize LLM client: {str(e)}{END}")
        return

    # Process facts using DKA approach
    print(f"\n{BOLD}🔍 Processing Facts with DKA Method...{END}")
    print(f"{CYAN}DKA relies on the model's internal knowledge without external retrieval{END}")

    results = []
    total_cost = 0.0
    total_tokens = 0
    successful_predictions = 0

    start_time = time.time()

    for i, fact_kg in enumerate(facts):
        identifier, fact = fact_kg
        print(f"\n{BLUE}Processing fact {i+1}/{len(facts)}{END}")
        print(f"  Fact: {fact.get('s', '')} {fact.get('p', '')} {fact.get('o', '')}")

        # Create DKA-specific prompt
        prompt = create_dka_prompt(fact, dataset_name)

        # Get LLM response
        response = llm_client.generate_response(prompt)

        # Process result
        result = {
            "id": identifier,
            "method": "DKA",
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
            print(f"  {GREEN}DKA Prediction: {predicted}{END}")

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

    # Print DKA summary
    print(f"\n{BOLD}{GREEN}📊 DKA Method Summary{END}")
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
    print(f"\n{BOLD}💾 Saving DKA Results...{END}")
    results_file = save_dka_results(results, config)

    if results_file:
        print(f"{GREEN}✅ DKA method completed successfully!{END}")
    else:
        print(f"{YELLOW}⚠️ DKA method completed but failed to save results{END}")