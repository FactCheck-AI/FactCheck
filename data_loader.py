import json
import logging
import os
from typing import Dict, List, Tuple, Any, Optional, Set

# Define colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
END = "\033[0m"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(
        dataset_name: str = "FactBench",
        dataset_file: str = "kg.json",
        kg_ids: Optional[List[str]] = None
) -> list[tuple[str, Any]]:
    """
    Enhanced dataset loading function with comprehensive error handling and filtering.

    Args:
        dataset_name: Name of the dataset (e.g., "FactBench", "DBpedia", "YAGO")
        dataset_file: Name of the knowledge graph file (default: "kg.json")
        kg_ids: Optional list of specific IDs to filter. If None, loads all data.

    Returns:
        Tuple containing:
        - kg: List of tuples (id, triple) representing knowledge graph facts
        - gt: Dictionary mapping fact IDs to ground truth labels

    Raises:
        FileNotFoundError: If dataset files are not found
        ValueError: If dataset format is invalid
        JSONDecodeError: If JSON files are malformed
    """

    print(f'{BOLD}{BLUE}📚 Loading {dataset_name} dataset...{END}')

    # Construct file paths
    dataset_dir = os.path.join('./dataset', dataset_name)
    kg_file_path = os.path.join(dataset_dir, 'data', dataset_file)
    gt_file_path = os.path.join(dataset_dir, 'data', 'gt.json')

    # Validate dataset directory exists
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Load knowledge graph data
    kg = _load_knowledge_graph(kg_file_path, dataset_name, gt_file_path, kg_ids)

    # Print summary statistics
    _print_dataset_summary(dataset_name, kg, kg_ids)

    return kg


def _load_knowledge_graph(
        kg_file_path: str,
        dataset_name: str,
        gt_file_path: str,
        kg_ids: Optional[List[str]] = None
) -> List[Tuple[str, Any]]:
    """
    Load and process knowledge graph data with optional ID filtering.

    Args:
        kg_file_path: Path to the knowledge graph JSON file
        dataset_name: Name of the dataset for special processing
        kg_ids: Optional list of IDs to filter

    Returns:
        List of tuples (id, triple) representing knowledge graph facts
    """

    print(f'  {CYAN}Loading knowledge graph from: {kg_file_path}{END}')

    # Load raw data
    try:
        modified_kg_file_path = None
        if 'modified' in kg_file_path:
            print(f'  {YELLOW}⚠️  Loading modified knowledge graph file: {kg_file_path}{END}')
            if os.path.exists(kg_file_path.replace('_modified', '')):
                modified_kg_file_path = kg_file_path
                kg_file_path = kg_file_path.replace('_modified', '')

        with open(kg_file_path, 'r', encoding='utf-8') as f:
            id2triple = json.load(f)
        if modified_kg_file_path:
            with open(modified_kg_file_path, 'r', encoding='utf-8') as f:
                id2triple_modified = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Knowledge graph file not found: {kg_file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in knowledge graph file: {e}")
    except Exception as e:
        raise Exception(f"Error loading knowledge graph: {str(e)}")

    if not isinstance(id2triple, dict):
        raise ValueError(f"Knowledge graph data should be a dictionary, got {type(id2triple)}")

    print(f'  {GREEN}✓ Loaded {len(id2triple)} raw facts{END}')

    # Convert to list of tuples
    kg = [(k, v) for k, v in id2triple.items()]

    kg_modified = {}
    if modified_kg_file_path:
        kg_modified = id2triple_modified

    # Apply dataset-specific filtering
    kg = _apply_dataset_filtering(kg, dataset_name)

    # Apply ID filtering if specified
    if kg_ids is not None:
        kg = _apply_id_filtering(kg, kg_ids)

    kg = _prepare_kg(kg, dataset_name, kg_modified)

    # Load ground truth data
    gt = _load_ground_truth(gt_file_path, kg)

    for fact_id, triple in kg:
        triple['label'] = gt[fact_id] if fact_id in gt else None
    return kg


def _apply_dataset_filtering(kg: List[Tuple[str, Any]], dataset_name: str) -> List[Tuple[str, Any]]:
    """
    Apply dataset-specific filtering rules.

    Args:
        kg: List of (id, triple) tuples
        dataset_name: Name of the dataset

    Returns:
        Filtered list of (id, triple) tuples
    """

    if dataset_name == "FactBench":
        print(f'  {YELLOW}Applying FactBench-specific filtering...{END}')

        # Extract first element if triple is a list/tuple
        kg = [(k, v[0] if isinstance(v, (list, tuple)) and len(v) > 0 else v) for k, v in kg]

        # Define FactBench fact type patterns
        factbench_patterns = [
            'correct_',
            'wrong_mix_domain',
            'wrong_mix_range',
            'wrong_mix_domainrange',
            'wrong_mix_property',
            'wrong_mix_random'
        ]

        # Filter based on patterns
        original_count = len(kg)
        kg = [p for p in kg if any(pattern in p[0] for pattern in factbench_patterns)]

        filtered_count = original_count - len(kg)
        print(f'  {GREEN}✓ FactBench filtering: kept {len(kg)} facts, filtered out {filtered_count}{END}')


    elif dataset_name in ["DBpedia", "YAGO"]:
        print(f'  {CYAN}No special filtering applied for {dataset_name}{END}')

    else:
        print(f'  {YELLOW}Unknown dataset {dataset_name}, no special filtering applied{END}')

    return kg


def _apply_id_filtering(kg: List[Tuple[str, Any]], kg_ids: List[str]) -> List[Tuple[str, Any]]:
    """
    Filter knowledge graph by specific IDs.

    Args:
        kg: List of (id, triple) tuples
        kg_ids: List of IDs to keep

    Returns:
        Filtered list of (id, triple) tuples
    """

    print(f'  {YELLOW}Filtering by {len(kg_ids)} specified IDs...{END}')

    # Convert to set for faster lookup
    kg_ids_set: Set[str] = set(kg_ids)

    # Filter knowledge graph
    original_count = len(kg)
    kg = [(k, v) for k, v in kg if k in kg_ids_set]

    # Check for missing IDs
    found_ids = {k for k, _ in kg}
    missing_ids = kg_ids_set - found_ids

    if missing_ids:
        print(f'  {RED}⚠️  Warning: {len(missing_ids)} requested IDs not found in dataset{END}')
        if len(missing_ids) <= 10:  # Show missing IDs if not too many
            print(f'  {RED}Missing IDs: {list(missing_ids)}{END}')
        else:
            print(f'  {RED}Missing IDs: {list(missing_ids)[:10]}... (and {len(missing_ids) - 10} more){END}')

    print(f'  {GREEN}✓ ID filtering: kept {len(kg)}/{original_count} facts{END}')

    return kg


def _load_ground_truth(gt_file_path: str, kg: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """
    Load ground truth labels and align with knowledge graph.

    Args:
        gt_file_path: Path to the ground truth JSON file
        kg: Knowledge graph list to align with

    Returns:
        Dictionary mapping fact IDs to ground truth labels
    """

    print(f'  {CYAN}Loading ground truth from: {gt_file_path}{END}')

    # Load ground truth data
    try:
        with open(gt_file_path, 'r', encoding='utf-8') as f:
            gt_raw = json.load(f)
    except FileNotFoundError:
        print(f'  {YELLOW}⚠️  Ground truth file not found: {gt_file_path}{END}')
        print(f'  {YELLOW}Proceeding without ground truth labels{END}')
        return {}
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in ground truth file: {e}")
    except Exception as e:
        raise Exception(f"Error loading ground truth: {str(e)}")

    if not isinstance(gt_raw, dict):
        raise ValueError(f"Ground truth data should be a dictionary, got {type(gt_raw)}")

    print(f'  {GREEN}✓ Loaded {len(gt_raw)} ground truth labels{END}')

    # Create set of knowledge graph IDs for efficient lookup
    kg_ids = {k[0] for k in kg}

    # Filter ground truth to only include facts present in knowledge graph
    gt = {k: v for k, v in gt_raw.items() if k in kg_ids}

    # Check alignment
    missing_gt = len(kg_ids) - len(gt)
    if missing_gt > 0:
        print(f'  {YELLOW}⚠️  {missing_gt} facts in KG have no ground truth labels{END}')

    extra_gt = len(gt_raw) - len(gt)
    if extra_gt > 0:
        print(f'  {YELLOW}⚠️  {extra_gt} ground truth labels have no corresponding facts in KG{END}')

    print(f'  {GREEN}✓ Aligned {len(gt)} ground truth labels with knowledge graph{END}')

    return gt


def _print_dataset_summary(
        dataset_name: str,
        kg: List[Tuple[str, Any]],
        kg_ids: Optional[List[str]]
) -> None:
    """Print a summary of the loaded dataset."""

    print(f'\n{BOLD}{GREEN}📊 Dataset Summary:{END}')
    print(f'  Dataset: {dataset_name}')
    print(f'  Total facts: {len(kg)}')

    if kg_ids is not None:
        print(f'  ID filtering: {len(kg_ids)} requested → {len(kg)} found')

    # Sample some facts
    if kg:
        print(f'\n{CYAN}Sample facts:{END}')
        sample_size = min(3, len(kg))
        for i in range(sample_size):
            fact_id, triple = kg[i]
            # gt_label = gt.get(fact_id, 'No label')
            print(f'  {i + 1}. ID: {fact_id}')
            print(f'     Triple: {triple}')
            # print(f'     Label: {gt_label}')

    print(f'{GREEN}✅ Dataset loaded successfully!{END}\n')


def validate_dataset_structure(dataset_name: str, dataset_file: str = "kg.json") -> bool:
    """
    Validate the structure of a dataset before loading.

    Args:
        dataset_name: Name of the dataset to validate
        dataset_file: Name of the knowledge graph file

    Returns:
        True if dataset structure is valid, False otherwise
    """

    print(f'{BOLD}{BLUE}🔍 Validating dataset structure for {dataset_name}...{END}')

    dataset_dir = os.path.join('./dataset', dataset_name)
    kg_file_path = os.path.join(dataset_dir, 'data', dataset_file)
    gt_file_path = os.path.join(dataset_dir, 'data', 'gt.json')

    issues = []

    # Check directory structure
    if not os.path.exists(dataset_dir):
        issues.append(f"Dataset directory missing: {dataset_dir}")

    if not os.path.exists(os.path.join(dataset_dir, 'data')):
        issues.append(f"Data directory missing: {os.path.join(dataset_dir, 'data')}")

    # Check required files
    if not os.path.exists(kg_file_path):
        issues.append(f"Knowledge graph file missing: {kg_file_path}")

    if not os.path.exists(gt_file_path):
        issues.append(f"Ground truth file missing: {gt_file_path}")

    # Validate JSON files
    if os.path.exists(kg_file_path):
        try:
            with open(kg_file_path, 'r', encoding='utf-8') as f:
                kg_data = json.load(f)
            if not isinstance(kg_data, dict):
                issues.append(f"Knowledge graph should be a JSON object, got {type(kg_data)}")
        except json.JSONDecodeError:
            issues.append(f"Knowledge graph file contains invalid JSON: {kg_file_path}")
        except Exception as e:
            issues.append(f"Error reading knowledge graph file: {str(e)}")

    if os.path.exists(gt_file_path):
        try:
            with open(gt_file_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
            if not isinstance(gt_data, dict):
                issues.append(f"Ground truth should be a JSON object, got {type(gt_data)}")
        except json.JSONDecodeError:
            issues.append(f"Ground truth file contains invalid JSON: {gt_file_path}")
        except Exception as e:
            issues.append(f"Error reading ground truth file: {str(e)}")

    # Print results
    if issues:
        print(f'{RED}❌ Dataset validation failed:{END}')
        for issue in issues:
            print(f'  {RED}• {issue}{END}')
        return False
    else:
        print(f'{GREEN}✅ Dataset structure is valid{END}')
        return True


def _prepare_kg(kg: List[Tuple[str, Any]], dataset_name: str, kg_modified: dict) -> List[Tuple[str, Any]]:
    """
    Prepare the knowledge graph by converting it to a list of tuples.

    Args:
        kg: List of (id, triple) tuples
        dataset_name: Name of the dataset

    Returns:
        Prepared list of (id, triple) tuples
    """

    print(f'  {CYAN}Preparing knowledge graph...{END}')

    prepared_kg = []
    for identifier, knowledge_graph in kg:
        if dataset_name == 'FactBench':
            s, p, o = knowledge_graph
        elif dataset_name in ['DBpedia']:
            s, p, o = map(lambda x: str(x).replace('_', ' '), knowledge_graph)
        else:
            # replace "_" with " " for YAGO and other datasets for all the elements in knowledge graph
            # Assuming knowledge_graph[1] is a tuple of (s, p, o)
            s, p, o = map(lambda x: str(x).replace('_', ' '), knowledge_graph)

        if kg_modified and identifier in kg_modified:
            prepared_kg.append((identifier, {'s': s, 'p': p, 'o': o, 'transformed': kg_modified[identifier]}))
        else:
            prepared_kg.append((identifier, {'s': s, 'p': p, 'o': o}))

    print(f'  {GREEN}✓ Knowledge graph prepared with {len(prepared_kg)} facts{END}')

    return prepared_kg


if __name__ == "__main__":
    load_dataset()
