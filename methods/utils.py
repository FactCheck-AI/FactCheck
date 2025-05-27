import json
import os
import time
from typing import Dict, List, Any, Optional

# Define colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
END = "\033[0m"


def save_results(mode: str, results: List[Dict[str, Any]], config: Dict[str, Any]) -> Optional[str]:
    """
    Save {mode} results to a file

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
        f'{dataset}_{llm_mode}_{llm_model}_{mode}_{time.strftime("%Y%m%d-%H%M%S")}.json'
    )

    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"{GREEN}✓ {mode.upper()} results saved to {results_file}{END}")
        return results_file
    except Exception as e:
        print(f"{RED}✗ Failed to save {mode.upper()} results: {str(e)}{END}")
        return None
