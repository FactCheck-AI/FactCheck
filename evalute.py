#!/usr/bin/env python3
"""
Evaluation script for FactCheck results using scikit-learn
"""

import argparse
import json
import os
import glob
import sys
from typing import Dict, List, Any
from collections import Counter
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Import the existing config module
from config import ConfigReader
from data_loader import load_dataset

# Define colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
END = "\033[0m"


def extract_dataset_from_filename(filename: str) -> str:
    """Extract dataset name from result filename"""
    # Remove extension and split by underscore
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_')

    if len(parts) >= 1:
        dataset_name = parts[0]
        print(f"{CYAN}📁 Detected dataset from filename: {dataset_name}{END}")
        return dataset_name
    else:
        print(f"{YELLOW}⚠️ Could not extract dataset from filename: {filename}{END}")
        return ""


def load_ground_truth(dataset_name: str) -> Dict[str, Any]:
    """Load ground truth data from dataset"""
    if not dataset_name:
        print(f"{RED}✗ No dataset name provided{END}")
        return {}

    try:
        gt = dict()
        full_kg = load_dataset(dataset_name=dataset_name, dataset_file='kg.json')
        for identifier, knowledge_graph in full_kg:
            gt[identifier] = knowledge_graph.get('label')
        return gt
    except Exception as e:
        print(f"{RED}✗ Invalid JSON in ground truth file: {e}{END}")
        return {}


def load_results_file(file_path: str) -> List[Dict[str, Any]]:
    """Load results from a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        print(f"{RED}✗ Results file not found: {file_path}{END}")
        return []
    except json.JSONDecodeError as e:
        print(f"{RED}✗ Invalid JSON in results file: {e}{END}")
        return []


def normalize_label(label: Any) -> str:
    """Normalize label to T/F format"""
    if isinstance(label, bool):
        return "T" if label else "F"
    elif isinstance(label, int):
        return "T" if label == 1 else "F"
    elif isinstance(label, str):
        label = label.strip().upper()
        if label in ['T', 'TRUE', '1']:
            return "T"
        elif label in ['F', 'FALSE', '0']:
            return "F"
    return str(label)


def normalize_prediction(response: str) -> str:
    """Normalize model response to T/F format"""
    if not response:
        return "UNKNOWN"

    response = response.strip().upper()

    # Direct matches
    if response in ['T', 'F']:
        return response
    elif response in ['TRUE', 'FALSE']:
        return 'T' if response == 'TRUE' else 'F'
    elif response.startswith('T'):
        return 'T'
    elif response.startswith('F'):
        return 'F'
    else:
        return "UNKNOWN"


def get_wrong_answer(correct_answer: str) -> str:
    """Get the opposite of the correct answer"""
    if correct_answer == "T":
        return "F"
    elif correct_answer == "F":
        return "T"
    else:
        return "F"  # Default to F if unclear


def convert_labels_to_binary(labels: List[str]) -> List[int]:
    """Convert T/F labels to binary for sklearn"""
    return [1 if label == "T" else 0 for label in labels]


def evaluate_results(results: List[Dict[str, Any]], config: Dict[str, Any],
                     dataset_name: str, full_evaluation: bool = False) -> Dict[str, Any]:
    """Evaluate results against ground truth using scikit-learn"""

    eval_config = config.get("evaluation", {}).get("metrics", {})
    accuracy_type = eval_config.get("accuracy", "balanced")
    f1_type = eval_config.get("f1_score", "macro")

    # Load ground truth for the detected dataset
    ground_truth = load_ground_truth(dataset_name)

    if not ground_truth:
        print(f"{RED}✗ No ground truth available for dataset: {dataset_name}{END}")
        return {}

    if full_evaluation:
        print(f"{CYAN}🔍 Running full evaluation against complete ground truth{END}")
        print(f"{YELLOW}   Missing predictions will be set to wrong answers{END}")

        # Create predictions for all ground truth items
        result_predictions = {r["id"]: normalize_prediction(r["response"]) for r in results if r.get("success", False)}

        y_true = []
        y_pred = []
        missing_count = 0

        for gt_id, gt_label in ground_truth.items():
            true_label = normalize_label(gt_label)
            y_true.append(true_label)

            # If we have a prediction, use it; otherwise, use wrong answer
            if gt_id in result_predictions:
                predicted = result_predictions[gt_id]
                if predicted == "UNKNOWN":
                    # Even successful predictions might be unclear, count as wrong
                    y_pred.append(get_wrong_answer(true_label))
                    missing_count += 1
                else:
                    y_pred.append(predicted)
            else:
                # Missing prediction - use wrong answer
                y_pred.append(get_wrong_answer(true_label))
                missing_count += 1

        total_items = len(ground_truth)
        evaluated_items = len(y_true)
        successful_predictions = len([p for p in result_predictions.values() if p != "UNKNOWN"])

        print(f"{CYAN}   Total ground truth items: {total_items}{END}")
        print(f"{CYAN}   Successful predictions: {successful_predictions}{END}")
        print(f"{CYAN}   Missing/unclear predictions: {missing_count}{END}")

    else:
        print(f"{CYAN}🔍 Running evaluation on successful predictions only{END}")
        # Only evaluate successful predictions with known ground truth
        y_true = []
        y_pred = []

        for result in results:
            if not result.get("success", False):
                continue

            # Get ground truth
            if result["id"] in ground_truth:
                gt_label = ground_truth[result["id"]]
            elif "label" in result.get("fact", {}):
                gt_label = result["fact"]["label"]
            else:
                continue  # Skip if no ground truth available

            predicted = normalize_prediction(result["response"])
            if predicted == "UNKNOWN":
                continue  # Skip unclear predictions in non-full mode

            y_true.append(normalize_label(gt_label))
            y_pred.append(predicted)

        total_items = len(results)
        evaluated_items = len(y_true)
        successful_predictions = evaluated_items

    if not y_true:
        print(f"{RED}✗ No data available for evaluation{END}")
        return {}

    # Convert to binary for sklearn (T=1, F=0)
    y_true_binary = convert_labels_to_binary(y_true)
    y_pred_binary = convert_labels_to_binary(y_pred)

    # Calculate metrics using sklearn
    if accuracy_type == "balanced":
        accuracy = balanced_accuracy_score(y_true_binary, y_pred_binary)
    else:
        accuracy = accuracy_score(y_true_binary, y_pred_binary)

    # Calculate precision, recall, F1 using sklearn
    precision, recall, f1_score, support = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average=f1_type, zero_division=0
    )

    # If macro/micro/weighted returns arrays, take the mean or appropriate value
    if isinstance(precision, np.ndarray):
        precision = float(precision.mean()) if len(precision) > 1 else float(precision[0])
    if isinstance(recall, np.ndarray):
        recall = float(recall.mean()) if len(recall) > 1 else float(recall[0])
    if isinstance(f1_score, np.ndarray):
        f1_score = float(f1_score.mean()) if len(f1_score) > 1 else float(f1_score[0])

    # Calculate confusion matrix using sklearn
    cm = confusion_matrix(y_true, y_pred, labels=["T", "F"])
    confusion_dict = {
        ("T", "T"): int(cm[0, 0]),
        ("T", "F"): int(cm[0, 1]),
        ("F", "T"): int(cm[1, 0]),
        ("F", "F"): int(cm[1, 1])
    }

    # Calculate additional statistics
    success_rate = sum(1 for r in results if r.get("success", False)) / len(results) if results else 0

    # Get detailed classification report
    class_report = classification_report(y_true, y_pred, labels=["T", "F"], output_dict=True, zero_division=0)

    evaluation_results = {
        "dataset": dataset_name,
        "total_items": total_items if full_evaluation else len(results),
        "evaluated_items": evaluated_items,
        "successful_predictions": successful_predictions,
        "success_rate": success_rate,
        "accuracy": float(accuracy),
        "accuracy_type": accuracy_type,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "f1_type": f1_type,
        "confusion_matrix": confusion_dict,
        "label_distribution": dict(Counter(y_true)),
        "prediction_distribution": dict(Counter(y_pred)),
        "classification_report": class_report
    }

    if full_evaluation:
        evaluation_results["missing_predictions"] = missing_count

    return evaluation_results


def print_evaluation_results(results: Dict[str, Any], file_name: str) -> None:
    """Print evaluation results in a formatted way"""

    print(f"\n{BOLD}{BLUE}📊 Evaluation Results for: {file_name}{END}")
    print(f"{BOLD}{BLUE}📚 Dataset: {results.get('dataset', 'Unknown')}{END}")
    print("=" * 80)

    # Basic statistics
    print(f"\n{BOLD}📈 Basic Statistics:{END}")
    print(f"  Total items: {results.get('total_items', 0)}")
    print(f"  Items evaluated: {results.get('evaluated_items', 0)}")
    print(f"  Successful predictions: {results.get('successful_predictions', 0)}")
    print(f"  Success rate: {results.get('success_rate', 0):.2%}")

    if "missing_predictions" in results:
        print(f"  Missing/unclear predictions: {results['missing_predictions']}")

    # Performance metrics
    print(f"\n{BOLD}🎯 Performance Metrics (using scikit-learn):{END}")
    print(f"  Accuracy ({results.get('accuracy_type', 'unknown')}): {GREEN}{results.get('accuracy', 0):.4f}{END}")
    print(f"  Precision ({results.get('f1_type', 'unknown')}): {CYAN}{results.get('precision', 0):.4f}{END}")
    print(f"  Recall ({results.get('f1_type', 'unknown')}): {CYAN}{results.get('recall', 0):.4f}{END}")
    print(f"  F1-Score ({results.get('f1_type', 'unknown')}): {MAGENTA}{results.get('f1_score', 0):.4f}{END}")

    # Confusion matrix
    confusion_matrix = results.get('confusion_matrix', {})
    if confusion_matrix:
        print(f"\n{BOLD}🔀 Confusion Matrix:{END}")
        true_pred_label = "True\\Pred"
        print(f"  {true_pred_label:<12} {'T':<8} {'F':<8}")
        print(f"  {'-' * 28}")

        for true_label in ['T', 'F']:
            row = f"  {true_label:<12}"
            for pred_label in ['T', 'F']:
                count = confusion_matrix.get((true_label, pred_label), 0)
                row += f" {count:<8}"
            print(row)

    # Per-class metrics from classification report
    class_report = results.get('classification_report', {})
    if class_report:
        print(f"\n{BOLD}📊 Per-Class Metrics:{END}")
        for label in ['T', 'F']:
            if label in class_report:
                metrics = class_report[label]
                print(f"  Class {label}:")
                print(f"    Precision: {metrics.get('precision', 0):.4f}")
                print(f"    Recall:    {metrics.get('recall', 0):.4f}")
                print(f"    F1-Score:  {metrics.get('f1-score', 0):.4f}")
                print(f"    Support:   {metrics.get('support', 0)}")

    # Label distribution
    label_dist = results.get('label_distribution', {})
    pred_dist = results.get('prediction_distribution', {})

    if label_dist:
        print(f"\n{BOLD}📊 Label Distribution:{END}")
        print(f"  Ground Truth: {dict(label_dist)}")
        print(f"  Predictions:  {dict(pred_dist)}")


def get_results_files(results_dir: str = "./results") -> List[str]:
    """Get all JSON result files from the results directory"""
    if not os.path.exists(results_dir):
        print(f"{RED}✗ Results directory not found: {results_dir}{END}")
        return []

    # Look for JSON files in results directory
    pattern = os.path.join(results_dir, "*.json")
    json_files = glob.glob(pattern)

    if not json_files:
        print(f"{YELLOW}⚠️ No JSON result files found in {results_dir}{END}")
        return []

    print(f"{GREEN}✓ Found {len(json_files)} result files{END}")
    return sorted(json_files)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluate FactCheck results using scikit-learn")
    parser.add_argument("--config", type=str, default="config.yml",
                        help="Path to configuration file")
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="Directory containing result files")
    parser.add_argument("--file", type=str,
                        required=True, help="Specific result file to evaluate")
    parser.add_argument("--full", action="store_true",
                        default=False, help="Evaluate against full ground truth (missing predictions set to wrong answers)")

    args = parser.parse_args()

    print(f"{BOLD}{CYAN}FactCheck Evaluation Tool (with scikit-learn){END}")
    print("-" * 50)

    # Load configuration using the existing config module
    try:
        config_reader = ConfigReader(args.config)
        config = config_reader.load_config()

        if not config:
            print(f"{YELLOW}⚠️ Failed to load configuration, proceeding with default settings{END}")
            config = {
                "evaluation": {
                    "metrics": {
                        "accuracy": "balanced",
                        "f1_score": "macro"
                    }
                }
            }
        else:
            print(f"{GREEN}✓ Configuration loaded successfully{END}")

    except Exception as e:
        print(f"{YELLOW}⚠️ Error loading configuration: {str(e)}{END}")
        print(f"{YELLOW}   Proceeding with default evaluation settings{END}")
        config = {
            "evaluation": {
                "metrics": {
                    "accuracy": "balanced",
                    "f1_score": "macro"
                }
            }
        }

    # Get result files
    if args.file:
        if os.path.exists(args.file):
            result_files = [args.file]
        else:
            print(f"{RED}✗ Specified file not found: {args.file}{END}")
            sys.exit(1)
    else:
        result_files = get_results_files(args.results_dir)

    if not result_files:
        print(f"{RED}✗ No result files to evaluate{END}")
        sys.exit(1)

    # Evaluate each file
    for file_path in result_files:
        file_name = os.path.basename(file_path)
        print(f"\n{BOLD}{YELLOW}📁 Processing: {file_name}{END}")

        # Extract dataset name from filename
        dataset_name = extract_dataset_from_filename(file_name)
        if not dataset_name:
            print(f"{RED}✗ Could not determine dataset from filename: {file_name}{END}")
            continue

        # Load results
        results = load_results_file(file_path)
        if not results:
            print(f"{RED}✗ Failed to load results from {file_name}{END}")
            continue

        # Evaluate results
        evaluation_results = evaluate_results(results, config, dataset_name, args.full)
        if not evaluation_results:
            print(f"{RED}✗ Failed to evaluate {file_name}{END}")
            continue

        # Print results
        print_evaluation_results(evaluation_results, file_name)

    print(f"\n{GREEN}✅ Evaluation completed!{END}")


if __name__ == "__main__":
    main()