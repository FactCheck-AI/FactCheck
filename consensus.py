"""
Result Merger for Majority Voting
Includes interactive file selection, consistency calculation, and advanced tie-breaking
"""

import json
import time
import os
import glob
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass
import argparse

from llm_client import LLMClient
from data_loader import load_dataset
from methods.dka import create_dka_prompt

# Define colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
END = "\033[0m"


@dataclass
class ModelResult:
    """Represents a single model's result for a fact"""
    model_name: str
    method: str
    response: str
    normalized_response: str
    success: bool
    response_time: float = 0.0
    timestamp: str = ""
    source_file: str = ""


@dataclass
class ConsistencyResult:
    """Represents model consistency statistics"""
    model_name: str
    consistency_score: float
    agreements: int
    total_facts: int
    source_file: str


@dataclass
class TieBreakerResult:
    """Represents results from all tie-breaker strategies"""
    fact_id: str
    commercial_result: Optional[str] = None
    commercial_model: Optional[str] = None
    commercial_response: Optional[str] = None
    most_consistent_result: Optional[str] = None
    most_consistent_model: Optional[str] = None
    most_consistent_response: Optional[str] = None
    least_consistent_result: Optional[str] = None
    least_consistent_model: Optional[str] = None
    least_consistent_response: Optional[str] = None
    strategies_attempted: List[str] = None
    strategies_successful: List[str] = None

    def __post_init__(self):
        if self.strategies_attempted is None:
            self.strategies_attempted = []
        if self.strategies_successful is None:
            self.strategies_successful = []


@dataclass
class MajorityVoteStats:
    """Statistics for majority voting process"""
    total_facts: int
    facts_with_majority: int
    ties_encountered: int
    ties_resolved: int
    tie_percentage: float
    decidable_facts: int
    consistency_scores: List[ConsistencyResult]
    commercial_ties_resolved: int = 0
    most_consistent_ties_resolved: int = 0
    least_consistent_ties_resolved: int = 0


@dataclass
class MergedResult:
    """Represents merged majority voting result"""
    fact_id: str
    fact_data: Dict[str, Any]
    individual_results: List[ModelResult]
    majority_decision: str
    confidence: float
    consensus_level: str
    tie_broken: bool = False
    tie_breaker_results: Optional[TieBreakerResult] = None
    final_tie_breaker_used: Optional[str] = None


class ResultMerger:
    """merger with interactive selection and consistency calculation"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mv_config = config.get("majority_vote", {})
        self.dataset_name = config.get("dataset", {}).get("name", "Unknown")
        self.majority_threshold = self.mv_config.get("num_votes", 3)

        # Initialize tie-breaker models
        self.tie_breaker_clients = {}
        self._initialize_tie_breakers()

        # Track tie-breaker results for separate file saving
        self.commercial_tie_results = []
        self.most_consistent_tie_results = []
        self.least_consistent_tie_results = []

    def _initialize_tie_breakers(self):
        """Initialize tie-breaker models"""
        print(f"{BLUE}🔧 Initializing tie-breaker models...{END}")

        # Commercial tie-breaker
        try:
            commercial_models = self.mv_config.get("commercial_model", ["gpt-4o-mini"])
            if commercial_models:
                tie_breaker_config = self.config.copy()
                tie_breaker_config["llm"]["model"] = commercial_models[0]
                tie_breaker_config["llm"]["mode"] = "commercial"
                client = LLMClient(tie_breaker_config)
                self.tie_breaker_clients["commercial"] = (commercial_models[0], client)
                print(f"  {GREEN}✓ Commercial tie-breaker: {commercial_models[0]}{END}")
        except Exception as e:
            print(f"  {YELLOW}⚠️ Commercial tie-breaker failed: {str(e)}{END}")

        # Higher parameter open-source tie-breakers
        higher_param_models = self.mv_config.get("higher_parameter_model", {})
        for base_model, higher_model in higher_param_models.items():
            try:
                tie_breaker_config = self.config.copy()
                tie_breaker_config["llm"]["model"] = higher_model
                tie_breaker_config["llm"]["mode"] = "open_source"
                client = LLMClient(tie_breaker_config)
                self.tie_breaker_clients[f"higher_{base_model}"] = (higher_model, client)
                print(f"  {GREEN}✓ Higher-param tie-breaker: {higher_model}{END}")
            except Exception as e:
                print(f"  {YELLOW}⚠️ Higher-param {higher_model} failed: {str(e)}{END}")

    def discover_dataset_files(self, results_dir: str, dataset_name: str) -> List[str]:
        """Discover files for a specific dataset"""
        if not os.path.exists(results_dir):
            print(f"{RED}✗ Results directory not found: {results_dir}{END}")
            return []

        # Look for files matching the dataset
        pattern = os.path.join(results_dir, f"{dataset_name}*.json")
        dataset_files = glob.glob(pattern)

        # Filter out merged/consensus files
        filtered_files = []
        for file_path in dataset_files:
            filename = os.path.basename(file_path)
            if not any(keyword in filename.lower() for keyword in ['majority', 'consensus', 'merged', 'tie']):
                filtered_files.append(file_path)

        return sorted(filtered_files)

    def interactive_file_selection(self, results_dir: str, dataset_name: str) -> List[str]:
        """Interactive file selection for dataset"""
        print(f"\n{BOLD}{BLUE}📁 Interactive File Selection{END}")
        print(f"Dataset: {dataset_name}")
        print("-" * 50)

        # Discover files
        available_files = self.discover_dataset_files(results_dir, dataset_name)

        if not available_files:
            print(f"{RED}✗ No files found for dataset: {dataset_name}{END}")
            return []

        print(f"{CYAN}Found {len(available_files)} files for {dataset_name}:{END}")
        for i, file_path in enumerate(available_files):
            filename = os.path.basename(file_path)

            # Try to extract model info and show file stats
            try:
                with open(file_path, 'r') as f:
                    results = json.load(f)
                    num_facts = len(results) if isinstance(results, list) else 0
                    success_rate = sum(1 for r in results if r.get('success', False)) / len(results) if results else 0

                # Extract model info from filename
                parts = filename.replace('.json', '').split('_')
                model_info = '_'.join(parts[1:3]) if len(parts) >= 3 else "unknown"
                method_info = parts[3] if len(parts) >= 4 else "unknown"

                print(f"  {i+1:2d}. {filename}")
                print(f"      Model: {model_info}, Method: {method_info}")
                print(f"      Facts: {num_facts}, Success: {success_rate:.1%}")

            except Exception as e:
                print(f"  {i+1:2d}. {filename} {RED}(error reading file){END}")

        # Get user input for number of files
        while True:
            try:
                max_files = len(available_files)
                num_files_input = input(f"\n{CYAN}How many files do you want to select? (1-{max_files}): {END}")
                num_files = int(num_files_input)

                if 1 <= num_files <= max_files:
                    break
                else:
                    print(f"{RED}Please enter a number between 1 and {max_files}{END}")
            except ValueError:
                print(f"{RED}Please enter a valid number{END}")

        # Get user file selection
        selected_files = []
        print(f"\n{CYAN}Select {num_files} file(s) by entering their numbers (e.g., 1 3 5):{END}")

        while len(selected_files) < num_files:
            try:
                remaining = num_files - len(selected_files)
                if remaining == 1:
                    selection_input = input(f"Enter 1 more file number: ")
                else:
                    selection_input = input(f"Enter {remaining} more file numbers (space-separated): ")

                # Parse selection
                selections = [int(x) for x in selection_input.split()]

                # Validate selections
                valid_selections = []
                for sel in selections:
                    if 1 <= sel <= len(available_files):
                        file_path = available_files[sel - 1]
                        if file_path not in selected_files:
                            valid_selections.append(file_path)
                        else:
                            print(f"{YELLOW}File {sel} already selected{END}")
                    else:
                        print(f"{RED}Invalid file number: {sel}{END}")

                selected_files.extend(valid_selections)

                if len(selected_files) > num_files:
                    selected_files = selected_files[:num_files]

            except ValueError:
                print(f"{RED}Please enter valid numbers separated by spaces{END}")

        print(f"\n{GREEN}✓ Selected {len(selected_files)} files:{END}")
        for file_path in selected_files:
            print(f"  - {os.path.basename(file_path)}")

        return selected_files

    def validate_file_paths(self, file_paths: List[str]) -> List[str]:
        """Validate that all file paths exist and are readable"""
        print(f"\n{BLUE}🔍 Validating file paths...{END}")

        valid_files = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"{RED}✗ File not found: {file_path}{END}")
                continue

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        valid_files.append(file_path)
                        print(f"{GREEN}✓ Valid: {os.path.basename(file_path)} ({len(data)} records){END}")
                    else:
                        print(f"{YELLOW}⚠️ Empty or invalid format: {os.path.basename(file_path)}{END}")
            except Exception as e:
                print(f"{RED}✗ Error reading {os.path.basename(file_path)}: {str(e)}{END}")

        print(f"\n{CYAN}Validation complete: {len(valid_files)}/{len(file_paths)} files valid{END}")
        return valid_files

    def extract_model_info_from_filename(self, filename: str) -> Tuple[str, str, str]:
        """Extract dataset, model, and method from filename"""
        base_name = os.path.splitext(os.path.basename(filename))[0]
        parts = base_name.split('_')

        if len(parts) >= 4:
            dataset = parts[0]
            mode = parts[1]
            model = parts[2]
            method = parts[3]

            return dataset, model.lower(), method

        # Fallback parsing
        dataset = parts[0] if parts else 'unknown'

        if 'qwen' in filename.lower():
            model = 'qwen2.5'
        elif 'gemma' in filename.lower():
            model = 'gemma2'
        elif 'llama' in filename.lower():
            model = 'llama3.1'
        elif 'mistral' in filename.lower():
            model = 'mistral'
        elif 'gpt' in filename.lower():
            model = 'gpt-4o-mini'
        else:
            model = 'unknown'

        method = 'dka' if 'dka' in filename.lower() else 'unknown'

        return dataset, model, method

    def load_and_merge_results(self, file_paths: List[str]) -> Dict[str, List[ModelResult]]:
        """Load and merge results from multiple files"""
        print(f"\n{BOLD}{BLUE}📊 Loading and merging results...{END}")

        merged_by_fact = defaultdict(list)
        file_stats = {}

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            dataset, model, method = self.extract_model_info_from_filename(filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)

                successful_results = 0
                for result in results:
                    if not result.get("success", False):
                        continue

                    fact_id = result.get("id", "")
                    if not fact_id:
                        continue

                    # Normalize response
                    normalized = self._normalize_response(result.get("response", ""))
                    if normalized == "UNKNOWN":
                        continue

                    # Create ModelResult
                    model_result = ModelResult(
                        model_name=model,
                        method=method,
                        response=result.get("response", ""),
                        normalized_response=normalized,
                        success=True,
                        response_time=result.get("response_time", 0.0),
                        timestamp=result.get("timestamp", ""),
                        source_file=filename
                    )

                    merged_by_fact[fact_id].append(model_result)
                    successful_results += 1

                file_stats[filename] = {
                    "model": model,
                    "method": method,
                    "total_results": len(results),
                    "successful_results": successful_results
                }

                print(f"  {GREEN}✓ {filename}: {successful_results}/{len(results)} successful{END}")

            except Exception as e:
                print(f"  {RED}✗ Error loading {filename}: {str(e)}{END}")

        print(f"\n{CYAN}Merge summary:{END}")
        print(f"  Total unique facts: {len(merged_by_fact)}")
        print(f"  Files processed: {len(file_stats)}")

        # Show fact coverage distribution
        coverage_dist = Counter(len(results) for results in merged_by_fact.values())
        print(f"\n{CYAN}Fact coverage distribution:{END}")
        for num_models, count in sorted(coverage_dist.items()):
            print(f"  {num_models} models: {count} facts")

        return dict(merged_by_fact), file_stats

    def calculate_consistency_scores(self, merged_results: Dict[str, List[ModelResult]]) -> List[ConsistencyResult]:
        """Calculate consistency scores for each model based on majority voting"""
        print(f"\n{BOLD}{BLUE}📈 Calculating consistency scores...{END}")

        # Find facts with enough models for majority voting
        majority_facts = {
            fact_id: results for fact_id, results in merged_results.items()
            if len(results) >= self.majority_threshold
        }

        if not majority_facts:
            print(f"{YELLOW}⚠️ No facts have enough models for majority voting{END}")
            return []

        # Calculate majority vote for each fact
        majority_decisions = {}
        for fact_id, model_results in majority_facts.items():
            votes = [result.normalized_response for result in model_results]
            vote_counts = Counter(votes)
            most_common = vote_counts.most_common(1)

            # Only include if there's a clear majority (>= majority_threshold)
            if most_common and most_common[0][1] >= self.majority_threshold:
                majority_decisions[fact_id] = most_common[0][0]

        print(f"  Facts with clear majority: {len(majority_decisions)}")

        # Calculate consistency for each model
        model_agreements = defaultdict(lambda: {"agreements": 0, "total": 0, "source_file": ""})

        for fact_id, majority_decision in majority_decisions.items():
            model_results = majority_facts[fact_id]

            for result in model_results:
                model_name = result.model_name
                model_agreements[model_name]["total"] += 1
                model_agreements[model_name]["source_file"] = result.source_file

                if result.normalized_response == majority_decision:
                    model_agreements[model_name]["agreements"] += 1

        # Create consistency results
        consistency_results = []
        for model_name, stats in model_agreements.items():
            if stats["total"] > 0:
                consistency_score = stats["agreements"] / stats["total"]
                consistency_result = ConsistencyResult(
                    model_name=model_name,
                    consistency_score=consistency_score,
                    agreements=stats["agreements"],
                    total_facts=stats["total"],
                    source_file=stats["source_file"]
                )
                consistency_results.append(consistency_result)

        # Sort by consistency score
        consistency_results.sort(key=lambda x: x.consistency_score, reverse=True)

        print(f"\n{CYAN}Model consistency scores:{END}")
        print(f"{'Model':<20} {'Consistency':<12} {'Agreements':<12} {'Total':<8} {'Source File'}")
        print("-" * 80)

        for result in consistency_results:
            bar_length = int(result.consistency_score * 20)  # Scale to 20 chars
            bar = "█" * bar_length

            print(f"{result.model_name:<20} {result.consistency_score:<12.4f} "
                  f"{result.agreements:<12} {result.total_facts:<8} {result.source_file}")
            print(f"{'':<20} {bar}")

        return consistency_results

    def apply_majority_voting_with_ties(self, merged_results: Dict[str, List[ModelResult]],
                                        consistency_scores: List[ConsistencyResult]) -> Tuple[List[MergedResult], MajorityVoteStats]:
        """Apply majority voting with comprehensive tie-breaking"""
        print(f"\n{BOLD}{BLUE}🗳️  Applying majority voting with comprehensive tie resolution...{END}")

        # Load ground truth facts for tie-breaking
        try:
            facts_dict = {}
            facts = load_dataset(dataset_name=self.dataset_name, dataset_file='kg.json')
            for fact_id, fact_data in facts:
                facts_dict[fact_id] = fact_data
        except Exception as e:
            print(f"{YELLOW}⚠️ Could not load fact data: {str(e)}{END}")
            facts_dict = {}

        # Get most and least consistent models for tie-breaking
        most_consistent = consistency_scores[0] if consistency_scores else None
        least_consistent = consistency_scores[-1] if consistency_scores else None

        majority_results = []
        stats = MajorityVoteStats(
            total_facts=len(merged_results),
            facts_with_majority=0,
            ties_encountered=0,
            ties_resolved=0,
            tie_percentage=0.0,
            decidable_facts=0,
            consistency_scores=consistency_scores,
            commercial_ties_resolved=0,
            most_consistent_ties_resolved=0,
            least_consistent_ties_resolved=0
        )

        for fact_id, model_results in merged_results.items():
            if len(model_results) < 2:
                continue  # Skip facts with only one model

            print(f"\n{BLUE}Processing fact: {fact_id}{END}")

            # Count votes
            votes = [result.normalized_response for result in model_results]
            vote_counts = Counter(votes)
            most_common = vote_counts.most_common()

            print(f"  Votes: {dict(vote_counts)}")

            tie_broken = False
            tie_breaker_results = None
            final_tie_breaker_used = None

            # Determine if we have a clear majority or tie
            if len(most_common) >= 2 and most_common[0][1] == most_common[1][1]:
                # Tie situation
                print(f"  {YELLOW}🔀 Tie detected!{END}")
                stats.ties_encountered += 1

                # Try ALL tie-breaking strategies comprehensively
                tie_breaker_results = self._resolve_tie_comprehensive(
                    fact_id, facts_dict.get(fact_id, {}),
                    most_consistent, least_consistent
                )

                # Determine which strategy to use as final decision
                final_decision, final_tie_breaker_used = self._choose_final_tie_breaker(self.config, tie_breaker_results)

                if final_decision:
                    majority_decision = final_decision
                    tie_broken = True
                    stats.ties_resolved += 1

                    # Update strategy-specific counters
                    if final_tie_breaker_used and "commercial" in final_tie_breaker_used:
                        stats.commercial_ties_resolved += 1
                    elif final_tie_breaker_used and "most_consistent" in final_tie_breaker_used:
                        stats.most_consistent_ties_resolved += 1
                    elif final_tie_breaker_used and "least_consistent" in final_tie_breaker_used:
                        stats.least_consistent_ties_resolved += 1

                    print(f"  {GREEN}✓ Tie resolved by {final_tie_breaker_used}: {majority_decision}{END}")
                else:
                    # Default to first vote
                    majority_decision = most_common[0][0]
                    print(f"  {RED}✗ All tie resolution strategies failed, using: {majority_decision}{END}")

                # Store tie results for separate file saving
                self._store_tie_results(fact_id, facts_dict.get(fact_id, {}), tie_breaker_results, model_results)

            else:
                # Clear majority
                majority_decision = most_common[0][0]
                stats.facts_with_majority += 1

            # Calculate confidence and consensus level
            if tie_broken:
                confidence = 1.0  # Tie-breaker provides full confidence
            else:
                confidence = most_common[0][1] / len(votes)

            consensus_level = self._get_consensus_level(vote_counts, len(votes))

            print(f"  {GREEN}Decision: {majority_decision} (confidence: {confidence:.2f}, {consensus_level}){END}")

            # Create merged result
            merged_result = MergedResult(
                fact_id=fact_id,
                fact_data=facts_dict.get(fact_id, {}),
                individual_results=model_results,
                majority_decision=majority_decision,
                confidence=confidence,
                consensus_level=consensus_level,
                tie_broken=tie_broken,
                tie_breaker_results=tie_breaker_results,
                final_tie_breaker_used=final_tie_breaker_used
            )

            majority_results.append(merged_result)
            stats.decidable_facts += 1

        # Calculate final statistics
        stats.tie_percentage = (stats.ties_encountered / stats.decidable_facts * 100) if stats.decidable_facts > 0 else 0.0

        print(f"\n{BOLD}{GREEN}📊 Majority Voting Statistics{END}")
        print(f"  Total facts processed: {stats.total_facts}")
        print(f"  Facts with majority decision: {stats.facts_with_majority}")
        print(f"  Ties encountered: {stats.ties_encountered}")
        print(f"  Ties resolved: {stats.ties_resolved}")
        print(f"  Commercial ties resolved: {stats.commercial_ties_resolved}")
        print(f"  Most consistent ties resolved: {stats.most_consistent_ties_resolved}")
        print(f"  Least consistent ties resolved: {stats.least_consistent_ties_resolved}")
        print(f"  Tie percentage: {stats.tie_percentage:.1f}%")
        if stats.ties_encountered > 0:
            print(f"  Tie resolution rate: {stats.ties_resolved/stats.ties_encountered:.1%}")

        return majority_results, stats

    def _resolve_tie_comprehensive(self, fact_id: str, fact_data: Dict[str, Any],
                                   most_consistent: Optional[ConsistencyResult],
                                   least_consistent: Optional[ConsistencyResult]) -> TieBreakerResult:
        """Tie resolution using ALL available strategies"""

        tie_result = TieBreakerResult(fact_id=fact_id)

        # Strategy 1: Commercial model
        if "commercial" in self.tie_breaker_clients:
            tie_result.strategies_attempted.append("commercial")
            model_name, client = self.tie_breaker_clients["commercial"]
            result = self._query_tie_breaker(client, fact_data)
            if result:
                tie_result.commercial_result = result
                tie_result.commercial_model = model_name
                tie_result.commercial_response = result
                tie_result.strategies_successful.append("commercial")
                print(f"    {GREEN}✓ Commercial ({model_name}): {result}{END}")
            else:
                print(f"    {RED}✗ Commercial ({model_name}): Failed{END}")

        # Strategy 2: Most consistent model (higher parameter)
        if most_consistent:
            tie_breaker_key = f"higher_{most_consistent.model_name}".lower()
            if tie_breaker_key in self.tie_breaker_clients:
                tie_result.strategies_attempted.append("most_consistent")
                model_name, client = self.tie_breaker_clients[tie_breaker_key]
                result = self._query_tie_breaker(client, fact_data)
                if result:
                    tie_result.most_consistent_result = result
                    tie_result.most_consistent_model = model_name
                    tie_result.most_consistent_response = result
                    tie_result.strategies_successful.append("most_consistent")
                    print(f"    {GREEN}✓ Most consistent ({model_name}): {result}{END}")
                else:
                    print(f"    {RED}✗ Most consistent ({model_name}): Failed{END}")

        # Strategy 3: Least consistent model (higher parameter)
        if least_consistent:
            tie_breaker_key = f"higher_{least_consistent.model_name}".lower()
            if tie_breaker_key in self.tie_breaker_clients:
                tie_result.strategies_attempted.append("least_consistent")
                model_name, client = self.tie_breaker_clients[tie_breaker_key]
                result = self._query_tie_breaker(client, fact_data)
                if result:
                    tie_result.least_consistent_result = result
                    tie_result.least_consistent_model = model_name
                    tie_result.least_consistent_response = result
                    tie_result.strategies_successful.append("least_consistent")
                    print(f"    {GREEN}✓ Least consistent ({model_name}): {result}{END}")
                else:
                    print(f"    {RED}✗ Least consistent ({model_name}): Failed{END}")

        return tie_result

    def _choose_final_tie_breaker(self, config, tie_result: TieBreakerResult) -> Tuple[Optional[str], Optional[str]]:
        """Choose which tie-breaker result to use as final decision"""
        # Priority order: Commercial > Most Consistent > Least Consistent -- if not specified, defaults to commercial

        majority_vote_config = config.get("majority_vote", {})
        tie_breaker_mode = majority_vote_config.get("mode", "commercial")
        final_tie_breaker = majority_vote_config.get("final_tie_breaker", "commercial")

        if final_tie_breaker not in ["commercial", "most_consistent", "least_consistent"]:
            if tie_breaker_mode in ["commercial"]:
                final_tie_breaker = "commercial"  # Default to commercial if invalid
            else:
                final_tie_breaker = "most_consistent"

        if tie_result.commercial_result and final_tie_breaker == "commercial":
            return tie_result.commercial_result, f"commercial_{tie_result.commercial_model}"
        elif tie_result.most_consistent_result and final_tie_breaker == "most_consistent":
            return tie_result.most_consistent_result, f"most_consistent_{tie_result.most_consistent_model}"
        elif tie_result.least_consistent_result and final_tie_breaker == "least_consistent":
            return tie_result.least_consistent_result, f"least_consistent_{tie_result.least_consistent_model}"
        else:
            return None, None

    def _store_tie_results(self, fact_id: str, fact_data: Dict[str, Any],
                           tie_result: TieBreakerResult, model_results: List[ModelResult]):
        """Store tie results for separate file saving"""

        base_tie_result = {
            "id": fact_id,
            "method": "TieBreaker",
            "fact": fact_data,
            "original_votes": [{"model": mr.model_name, "response": mr.normalized_response} for mr in model_results],
            "success": True,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Store commercial tie result
        if tie_result.commercial_result:
            commercial_result = base_tie_result.copy()
            commercial_result.update({
                "tie_breaker_model": tie_result.commercial_model,
                "response": tie_result.commercial_result,
                "tie_breaker_type": "commercial"
            })
            self.commercial_tie_results.append(commercial_result)

        # Store most consistent tie result
        if tie_result.most_consistent_result:
            most_consistent_result = base_tie_result.copy()
            most_consistent_result.update({
                "tie_breaker_model": tie_result.most_consistent_model,
                "response": tie_result.most_consistent_result,
                "tie_breaker_type": "most_consistent"
            })
            self.most_consistent_tie_results.append(most_consistent_result)

        # Store least consistent tie result
        if tie_result.least_consistent_result:
            least_consistent_result = base_tie_result.copy()
            least_consistent_result.update({
                "tie_breaker_model": tie_result.least_consistent_model,
                "response": tie_result.least_consistent_result,
                "tie_breaker_type": "least_consistent"
            })
            self.least_consistent_tie_results.append(least_consistent_result)

    def _query_tie_breaker(self, client: LLMClient, fact_data: Dict[str, Any]) -> Optional[str]:
        """Query a tie-breaker model"""
        try:
            prompt = create_dka_prompt(fact_data, self.dataset_name)
            response = client.generate_response(prompt)

            if response.success:
                normalized = self._normalize_response(response.content)
                if normalized != "UNKNOWN":
                    return normalized
        except Exception as e:
            print(f"    {RED}Tie-breaker error: {str(e)}{END}")

        return None

    def _normalize_response(self, response: str) -> str:
        """Normalize model response to T/F format"""
        if not response:
            return "UNKNOWN"

        response = response.strip().upper()

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

    def _get_consensus_level(self, vote_counts: Counter, total_votes: int) -> str:
        """Determine consensus level"""
        max_votes = max(vote_counts.values())

        if max_votes == total_votes:
            return "unanimous"
        elif max_votes >= (total_votes * 0.75):
            return "strong_majority"
        elif max_votes > (total_votes / 2):
            return "weak_majority"
        else:
            return "tie"

    def save_separate_tie_files(self, output_dir: str = "./results") -> Dict[str, Optional[str]]:
        """Save separate files for each tie-breaker strategy"""
        saved_files = {
            "commercial": None,
            "most_consistent": None,
            "least_consistent": None
        }

        timestamp = time.strftime('%Y%m%d-%H%M%S')

        # Save commercial tie results
        if self.commercial_tie_results:
            filename = f"{self.dataset_name}_tie-commercial_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.commercial_tie_results, f, indent=4, ensure_ascii=False)
                saved_files["commercial"] = filepath
                print(f"{GREEN}✓ Commercial tie results saved to {filename}{END}")
            except Exception as e:
                print(f"{RED}✗ Failed to save commercial tie results: {str(e)}{END}")

        # Save most consistent tie results
        if self.most_consistent_tie_results:
            filename = f"{self.dataset_name}_tie-most-consistent_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.most_consistent_tie_results, f, indent=4, ensure_ascii=False)
                saved_files["most_consistent"] = filepath
                print(f"{GREEN}✓ Most consistent tie results saved to {filename}{END}")
            except Exception as e:
                print(f"{RED}✗ Failed to save most consistent tie results: {str(e)}{END}")

        # Save least consistent tie results
        if self.least_consistent_tie_results:
            filename = f"{self.dataset_name}_tie-least-consistent_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.least_consistent_tie_results, f, indent=4, ensure_ascii=False)
                saved_files["least_consistent"] = filepath
                print(f"{GREEN}✓ Least consistent tie results saved to {filename}{END}")
            except Exception as e:
                print(f"{RED}✗ Failed to save least consistent tie results: {str(e)}{END}")

        return saved_files

    def save_comprehensive_results(self, merged_results: List[MergedResult],
                                   stats: MajorityVoteStats,
                                   file_paths: List[str],
                                   output_dir: str = "./results") -> Optional[str]:
        """Save comprehensive results with all statistics"""
        os.makedirs(output_dir, exist_ok=True)

        # Prepare results
        comprehensive_results = {
            "metadata": {
                "dataset": self.dataset_name,
                "source_files": [os.path.basename(f) for f in file_paths],
                "majority_threshold": self.majority_threshold,
                "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_facts": stats.total_facts,
                "decidable_facts": stats.decidable_facts
            },
            "statistics": {
                "facts_with_majority": stats.facts_with_majority,
                "ties_encountered": stats.ties_encountered,
                "ties_resolved": stats.ties_resolved,
                "commercial_ties_resolved": stats.commercial_ties_resolved,
                "most_consistent_ties_resolved": stats.most_consistent_ties_resolved,
                "least_consistent_ties_resolved": stats.least_consistent_ties_resolved,
                "tie_percentage": stats.tie_percentage,
                "tie_resolution_rate": (stats.ties_resolved / stats.ties_encountered * 100) if stats.ties_encountered > 0 else 0.0
            },
            "consistency_scores": [
                {
                    "model_name": cs.model_name,
                    "consistency_score": cs.consistency_score,
                    "agreements": cs.agreements,
                    "total_facts": cs.total_facts,
                    "source_file": cs.source_file
                }
                for cs in stats.consistency_scores
            ],
            "results": []
        }

        # Add individual results
        for result in merged_results:
            # Convert individual results
            individual_votes = []
            for model_result in result.individual_results:
                individual_votes.append({
                    "model_name": model_result.model_name,
                    "method": model_result.method,
                    "response": model_result.response,
                    "normalized_response": model_result.normalized_response,
                    "success": model_result.success,
                    "response_time": model_result.response_time,
                    "timestamp": model_result.timestamp,
                    "source_file": model_result.source_file
                })

            # Convert tie breaker results
            tie_breaker_info = None
            if result.tie_breaker_results:
                tie_breaker_info = {
                    "strategies_attempted": result.tie_breaker_results.strategies_attempted,
                    "strategies_successful": result.tie_breaker_results.strategies_successful,
                    "commercial_result": result.tie_breaker_results.commercial_result,
                    "commercial_model": result.tie_breaker_results.commercial_model,
                    "most_consistent_result": result.tie_breaker_results.most_consistent_result,
                    "most_consistent_model": result.tie_breaker_results.most_consistent_model,
                    "least_consistent_result": result.tie_breaker_results.least_consistent_result,
                    "least_consistent_model": result.tie_breaker_results.least_consistent_model,
                    "final_tie_breaker_used": result.final_tie_breaker_used
                }

            result_dict = {
                "id": result.fact_id,
                "method": "MajorityVote",
                "fact": result.fact_data,
                "majority_decision": result.majority_decision,
                "confidence": result.confidence,
                "consensus_level": result.consensus_level,
                "tie_broken": result.tie_broken,
                "tie_breaker_info": tie_breaker_info,
                "individual_votes": individual_votes,
                "success": result.majority_decision != "UNKNOWN"
            }

            comprehensive_results["results"].append(result_dict)

        # Create filename
        filename = f"{self.dataset_name}_majority-vote_{time.strftime('%Y%m%d-%H%M%S')}.json"
        filepath = os.path.join(output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_results, f, indent=4, ensure_ascii=False)

            print(f"\n{GREEN}✓ Comprehensive results saved to {filepath}{END}")

            # Also save separate tie files
            print(f"\n{BLUE}💾 Saving separate tie-breaker files...{END}")
            tie_files = self.save_separate_tie_files(output_dir)

            return filepath

        except Exception as e:
            print(f"{RED}✗ Failed to save results: {str(e)}{END}")
            return None


def main():
    """main function with interactive features"""
    parser = argparse.ArgumentParser(description="Majority Voting with Tie Resolution")
    parser.add_argument("--config", type=str, default="config.yml",
                        help="Path to configuration file")
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="Directory containing result files")
    parser.add_argument("--dataset", type=str,
                        help="Filter by dataset name for interactive selection")
    parser.add_argument("--files", nargs="+",
                        help="Specific files to merge")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Output directory for merged results")

    args = parser.parse_args()

    print(f"{BOLD}{CYAN}Result Merger for Majority Voting with Comprehensive Tie Resolution{END}")
    print("=" * 70)

    # Load configuration
    try:
        from config import ConfigReader
        config_reader = ConfigReader(args.config)
        config = config_reader.load_config()

        if not config:
            print(f"{YELLOW}⚠️ Using default configuration{END}")
            config = {
                "dataset": {"name": args.dataset or "FactBench"},
                "majority_vote": {
                    "num_votes": 3,
                    "commercial_model": ["gpt-4o-mini"],
                    "higher_parameter_model": {
                        "gemma2:9b": "gemma2:27b",
                        "qwen2.5:7b": "qwen2.5:14b",
                        "llama3.1:8b": "llama3.1:70b",
                        "mistral:7b": "mistral:nemo:12b"
                    }
                }
            }

        # Override dataset if specified
        if args.dataset:
            config["dataset"]["name"] = args.dataset

    except Exception as e:
        print(f"{YELLOW}⚠️ Configuration error: {str(e)}{END}")
        config = {
            "dataset": {"name": args.dataset or "FactBench"},
            "majority_vote": {"num_votes": 3}
        }

    # Initialize merger
    merger = ResultMerger(config)

    # Get file paths
    if args.files:
        # Validate specified files
        file_paths = merger.validate_file_paths(args.files)
    elif args.dataset:
        # Interactive selection
        file_paths = merger.interactive_file_selection(args.results_dir, args.dataset)
    else:
        print(f"{RED}✗ Either --dataset or --files must be specified{END}")
        return

    if not file_paths:
        print(f"{RED}✗ No valid files to process{END}")
        return

    # Load and merge results
    merged_by_fact, file_stats = merger.load_and_merge_results(file_paths)

    if not merged_by_fact:
        print(f"{RED}✗ No facts to process after merging{END}")
        return

    # Calculate consistency scores
    consistency_scores = merger.calculate_consistency_scores(merged_by_fact)

    # Apply majority voting with comprehensive tie-breaking
    majority_results, stats = merger.apply_majority_voting_with_ties(merged_by_fact, consistency_scores)

    # Save results
    output_file = merger.save_comprehensive_results(
        majority_results, stats, file_paths, args.output_dir
    )

    if output_file:
        print(f"\n{BOLD}{GREEN}🎉 Majority Voting with Tie Resolution Completed Successfully!{END}")
        print(f"Main output file: {output_file}")
        print(f"\nSeparate tie-breaker files have been saved for each strategy.")
        print(f"\nTo evaluate the results, run:")
        print(f"python evaluation.py --file {output_file}")
    else:
        print(f"\n{YELLOW}⚠️ Majority voting completed but failed to save results{END}")


if __name__ == "__main__":
    main()