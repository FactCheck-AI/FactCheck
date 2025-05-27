import sys
from dataclasses import dataclass
from typing import Dict, List, Any

import requests
import yaml

# Define colors for terminal output
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
END = "\033[0m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class ConfigValidator:
    """Validates configuration parameters"""

    # Valid options for different configuration sections
    VALID_DATASETS = ["DBpedia", "YAGO", "FactBench"]
    VALID_METHODS = ["DKA", "GIV-Z", "GIV-F", "RAG"]
    VALID_MODES = ["commercial", "open_source"]
    VALID_CHUNKING_STRATEGIES = ["fixed_size", "sliding_window", "small2big"]
    VALID_ACCURACY_TYPES = ["balanced", "normal"]
    VALID_F1_TYPES = ["micro", "macro", "weighted"]
    VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    # Known commercial models
    COMMERCIAL_MODELS = [
        "gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo",
        "DeepSeek-V3", "DeepSeek-R1", "claude-3-sonnet", "claude-3-haiku"
    ]

    # Known open source models (Ollama format)
    OLLAMA_MODELS = [
        "qwen2.5:7b", "llama3.1:8b", "gemma2:9b", "mistral:7b", "qwq:32b",
        "llama3:8b", "mistral:latest", "codellama:7b", "vicuna:7b"
    ]

    def __init__(self):
        self.ollama_running = False
        self.available_ollama_models = []
        self._check_ollama_status()

    def _check_ollama_status(self) -> None:
        """Check if Ollama is running and get available models"""
        try:
            # Check if Ollama service is running
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                self.ollama_running = True
                models_data = response.json()
                self.available_ollama_models = [
                    model["name"] for model in models_data.get("models", [])
                ]
                print(f"{GREEN}✓ Ollama is running{END}")
                print(f"{CYAN}  Available models: {', '.join(self.available_ollama_models)}{END}")
            else:
                print(f"{RED}✗ Ollama API returned status code: {response.status_code}{END}")
        except requests.RequestException as e:
            print(f"{RED}✗ Ollama is not running or not accessible: {str(e)}{END}")
            print(f"{YELLOW}  To start Ollama, run: ollama serve{END}")
        except Exception as e:
            print(f"{RED}✗ Error checking Ollama status: {str(e)}{END}")

    def validate_dataset_config(self, dataset_config: Dict[str, Any]) -> List[str]:
        """Validate dataset configuration"""
        errors = []

        if "name" not in dataset_config:
            errors.append("Dataset name is required")
        elif dataset_config["name"] not in self.VALID_DATASETS:
            errors.append(f"Invalid dataset name: {dataset_config['name']}. "
                          f"Valid options: {', '.join(self.VALID_DATASETS)}")

        return errors

    def validate_method_config(self, method_config: Dict[str, Any]) -> List[str]:
        """Validate method configuration"""
        errors = []

        if "name" not in method_config:
            errors.append("Method name is required")
        elif method_config["name"] not in self.VALID_METHODS:
            errors.append(f"Invalid method name: {method_config['name']}. "
                          f"Valid options: {', '.join(self.VALID_METHODS)}")

        return errors

    def validate_llm_config(self, llm_config: Dict[str, Any]) -> List[str]:
        """Validate LLM configuration"""
        errors = []
        warnings = []

        # Validate mode
        if "mode" not in llm_config:
            errors.append("LLM mode is required")
            return errors

        mode = llm_config["mode"]
        if mode not in self.VALID_MODES:
            errors.append(f"Invalid LLM mode: {mode}. Valid options: {', '.join(self.VALID_MODES)}")
            return errors

        # Validate model
        if "model" not in llm_config:
            errors.append("LLM model is required")
            return errors

        model = llm_config["model"].lower()

        if mode == "open_source":
            if not self.ollama_running:
                warnings.append("Ollama is not running. Open source models may not work.")
            elif model not in [m.lower() for m in self.available_ollama_models]:
                warnings.append(f"Model '{llm_config['model']}' not found in Ollama. "
                                f"Available models: {', '.join(self.available_ollama_models)}")
        elif mode == "commercial":
            if model not in [m.lower() for m in self.COMMERCIAL_MODELS]:
                warnings.append(f"Model '{llm_config['model']}' might not be a known commercial model")

        # Validate parameters
        if "parameters" in llm_config:
            params = llm_config["parameters"]
            if "temperature" in params:
                temp = params["temperature"]
                if not (0 <= temp <= 2):
                    errors.append("Temperature should be between 0 and 2")

            if "top_p" in params:
                top_p = params["top_p"]
                if not (0 <= top_p <= 1):
                    errors.append("top_p should be between 0 and 1")

            if "max_tokens" in params:
                max_tokens = params["max_tokens"]
                if not (1 <= max_tokens <= 4096):
                    warnings.append("max_tokens typically should be between 1 and 4096")

        return errors + warnings

    def validate_majority_vote_config(self, mv_config: Dict[str, Any]) -> List[str]:
        """Validate majority vote configuration"""
        errors = []
        warnings = []

        if "mode" in mv_config and mv_config["mode"] not in self.VALID_MODES:
            errors.append(f"Invalid majority vote mode: {mv_config['mode']}")
            return errors

        mode = mv_config.get("mode", "")
        num_votes = mv_config.get("num_votes", 0)

        if "num_votes" in mv_config:
            if not isinstance(num_votes, int) or num_votes < 1:
                errors.append("num_votes should be a positive integer")

        # Validate based on mode
        if mode == "commercial":
            # Check if commercial_model list exists and models are valid
            commercial_models = mv_config.get("commercial_model", [])
            if not commercial_models:
                errors.append("commercial_model list is required when mode is 'commercial'")
            else:
                for model in commercial_models:
                    model_lower = model.lower()
                    if model_lower not in [m.lower() for m in self.COMMERCIAL_MODELS]:
                        warnings.append(f"Commercial model '{model}' might not be available")

        elif mode == "open_source":
            llms = mv_config.get("llms", [])
            higher_param_models = mv_config.get("higher_parameter_model", {})

            # Check if we have enough models for voting
            if len(llms) < num_votes:
                errors.append(
                    f"Number of LLMs ({len(llms)}) should be greater than or equal to num_votes ({num_votes})")

            # Check if each model has a corresponding higher parameter model
            if not higher_param_models:
                errors.append("higher_parameter_model mapping is required when mode is 'open_source'")
            else:
                for model in llms:
                    model_key = model.lower()
                    if model_key not in higher_param_models:
                        errors.append(f"Model '{model}' missing from higher_parameter_model mapping")
                    else:
                        higher_model = higher_param_models[model_key]
                        # Check if the higher parameter model is available in Ollama
                        if self.ollama_running and higher_model.lower() not in [m.lower() for m in
                                                                                self.available_ollama_models]:
                            warnings.append(
                                f"Higher parameter model '{higher_model}' for '{model}' not found in Ollama")

            # Check if base models are available in Ollama
            if "llms" in mv_config:
                for model in llms:
                    model_lower = model.lower()
                    if self.ollama_running and model_lower not in [m.lower() for m in self.available_ollama_models]:
                        warnings.append(f"Majority vote model '{model}' not found in Ollama")

        return errors + warnings

    def validate_rag_config(self, rag_config: Dict[str, Any]) -> List[str]:
        """Validate RAG configuration"""
        errors = []

        if "chunking_strategy" in rag_config:
            strategy = rag_config["chunking_strategy"]
            if strategy not in self.VALID_CHUNKING_STRATEGIES:
                errors.append(f"Invalid chunking strategy: {strategy}. "
                              f"Valid options: {', '.join(self.VALID_CHUNKING_STRATEGIES)}")

        if "chunk_size" in rag_config:
            chunk_size = rag_config["chunk_size"]
            if not isinstance(chunk_size, int) or chunk_size < 1:
                errors.append("chunk_size should be a positive integer")

        if "similarity_cutoff" in rag_config:
            cutoff = rag_config["similarity_cutoff"]
            if not (0 <= cutoff <= 1):
                errors.append("similarity_cutoff should be between 0 and 1")

        if "top_k" in rag_config:
            top_k = rag_config["top_k"]
            if not isinstance(top_k, int) or top_k < 1:
                errors.append("top_k should be a positive integer")

        return errors

    def validate_evaluation_config(self, eval_config: Dict[str, Any]) -> List[str]:
        """Validate evaluation configuration"""
        errors = []

        if "metrics" in eval_config:
            metrics = eval_config["metrics"]

            if "accuracy" in metrics:
                accuracy = metrics["accuracy"]
                if accuracy not in self.VALID_ACCURACY_TYPES:
                    errors.append(f"Invalid accuracy type: {accuracy}. "
                                  f"Valid options: {', '.join(self.VALID_ACCURACY_TYPES)}")

            if "f1_score" in metrics:
                f1_score = metrics["f1_score"]
                if f1_score not in self.VALID_F1_TYPES:
                    errors.append(f"Invalid f1_score type: {f1_score}. "
                                  f"Valid options: {', '.join(self.VALID_F1_TYPES)}")

        return errors

    def validate_openai_config(self, openai_config: Dict[str, Any]) -> List[str]:
        """Validate OpenAI configuration"""
        warnings = []

        required_fields = ["azure_endpoint", "api_key", "api_version"]
        for field in required_fields:
            if field not in openai_config or not openai_config[field]:
                warnings.append(f"OpenAI {field} is empty - required for commercial models")

        return warnings

    def validate_logging_config(self, logging_config: Dict[str, Any]) -> List[str]:
        """Validate logging configuration"""
        errors = []

        if "level" in logging_config:
            level = logging_config["level"]
            if level not in self.VALID_LOG_LEVELS:
                errors.append(f"Invalid logging level: {level}. "
                              f"Valid options: {', '.join(self.VALID_LOG_LEVELS)}")

        return errors


class ConfigReader:
    """Reads and validates configuration from YAML file"""

    def __init__(self, config_path: str = "config.yml"):
        self.config_path = config_path
        self.config = {}
        self.validator = ConfigValidator()
        self.validation_result = ValidationResult(True, [], [])

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
                print(f"{GREEN}✓ Configuration loaded successfully from {self.config_path}{END}")
                return self.config
        except FileNotFoundError:
            error_msg = f"Configuration file {self.config_path} not found"
            print(f"{RED}✗ {error_msg}{END}")
            self.validation_result.errors.append(error_msg)
            return {}
        except yaml.YAMLError as e:
            error_msg = f"Error parsing YAML file: {str(e)}"
            print(f"{RED}✗ {error_msg}{END}")
            self.validation_result.errors.append(error_msg)
            return {}

    def validate_config(self) -> ValidationResult:
        """Validate the loaded configuration"""
        if not self.config:
            return ValidationResult(False, ["No configuration loaded"], [])

        all_errors = []
        all_warnings = []

        # Get method name to determine which sections to validate
        method_name = self.config.get("method", {}).get("name", "")

        # Validate each section
        if "dataset" in self.config:
            errors = self.validator.validate_dataset_config(self.config["dataset"])
            all_errors.extend(errors)

        if "method" in self.config:
            errors = self.validator.validate_method_config(self.config["method"])
            all_errors.extend(errors)

        if "llm" in self.config:
            messages = self.validator.validate_llm_config(self.config["llm"])
            # Separate errors from warnings (errors typically contain "Error" or specific validation failures)
            for msg in messages:
                if any(word in msg.lower() for word in ["error", "required", "should be between"]):
                    all_errors.append(msg)
                else:
                    all_warnings.append(msg)

        if "majority_vote" in self.config:
            messages = self.validator.validate_majority_vote_config(self.config["majority_vote"])
            for msg in messages:
                if "should be" in msg.lower() or "required" in msg.lower() or "missing" in msg.lower():
                    all_errors.append(msg)
                else:
                    all_warnings.append(msg)

        # Only validate RAG configuration if method is "RAG"
        if "rag" in self.config and method_name.upper() == "RAG":
            errors = self.validator.validate_rag_config(self.config["rag"])
            all_errors.extend(errors)
        elif "rag" in self.config and method_name.upper() != "RAG":
            all_warnings.append(f"RAG configuration found but method is '{method_name}' - RAG config will be ignored")

        if "evaluation" in self.config:
            errors = self.validator.validate_evaluation_config(self.config["evaluation"])
            all_errors.extend(errors)

        if "OpenAI" in self.config:
            warnings = self.validator.validate_openai_config(self.config["OpenAI"])
            all_warnings.extend(warnings)

        if "logging" in self.config:
            errors = self.validator.validate_logging_config(self.config["logging"])
            all_errors.extend(errors)

        self.validation_result = ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings
        )

        return self.validation_result

    def print_validation_results(self) -> None:
        """Print validation results"""
        print(f"\n{BOLD}{UNDERLINE}Validation Results:{END}")

        if self.validation_result.errors:
            print(f"\n{RED}{BOLD}Errors:{END}")
            for i, error in enumerate(self.validation_result.errors, 1):
                print(f"  {RED}{i}. {error}{END}")

        if self.validation_result.warnings:
            print(f"\n{YELLOW}{BOLD}Warnings:{END}")
            for i, warning in enumerate(self.validation_result.warnings, 1):
                print(f"  {YELLOW}{i}. {warning}{END}")

        if not self.validation_result.errors and not self.validation_result.warnings:
            print(f"{GREEN}✓ All validations passed!{END}")

        status = "VALID" if self.validation_result.is_valid else "INVALID"
        color = GREEN if self.validation_result.is_valid else RED
        print(f"\n{BOLD}Configuration Status: {color}{status}{END}")

    def print_configuration(self) -> None:
        """Beautifully print the configuration"""
        if not self.config:
            print(f"{RED}No configuration to display{END}")
            return

        # Get method name to determine which sections to display
        method_name = self.config.get("method", {}).get("name", "").upper()

        # Calculate width for consistent formatting
        width = 100

        # Print header
        print("\n" + "=" * width)
        print(f"{BOLD}{'FactCheck Experiment Configuration':^{width}}{END}")
        print("=" * width)

        # Dataset Configuration
        if "dataset" in self.config:
            print(f"\n{BOLD}{BLUE}📊 Dataset Configuration:{END}")
            dataset = self.config["dataset"]
            print(f"  {YELLOW}Dataset:{END} {GREEN}{dataset.get('name', 'Not specified')}{END}")

        # Method Configuration
        if "method" in self.config:
            print(f"\n{BOLD}{BLUE}🔬 Method Configuration:{END}")
            method = self.config["method"]
            print(f"  {YELLOW}Method:{END} {GREEN}{method.get('name', 'Not specified')}{END}")

        # LLM Configuration
        if "llm" in self.config:
            print(f"\n{BOLD}{BLUE}🤖 LLM Configuration:{END}")
            llm = self.config["llm"]
            mode = llm.get("mode", "Not specified")
            model = llm.get("model", "Not specified")
            model_type = "🏢 Commercial" if mode == "commercial" else "🔓 Open Source"
            print(f"  {YELLOW}Mode:{END} {GREEN}{mode}{END} ({model_type})")
            print(f"  {YELLOW}Model:{END} {GREEN}{model}{END}")

            if "parameters" in llm:
                params = llm["parameters"]
                print(f"  {YELLOW}Parameters:{END}")
                print(f"    {CYAN}Temperature:{END} {params.get('temperature', 'Not set')}")
                print(f"    {CYAN}Top P:{END} {params.get('top_p', 'Not set')}")
                print(f"    {CYAN}Max Tokens:{END} {params.get('max_tokens', 'Not set')}")

        # Majority Vote Configuration
        if "majority_vote" in self.config:
            print(f"\n{BOLD}{BLUE}🗳️  Majority Vote Configuration:{END}")
            mv = self.config["majority_vote"]
            mv_mode = mv.get('mode', 'Not specified')
            print(f"  {YELLOW}Mode:{END} {GREEN}{mv_mode}{END}")
            print(f"  {YELLOW}Number of Votes:{END} {GREEN}{mv.get('num_votes', 'Not specified')}{END}")

            if mv_mode == "open_source" and "llms" in mv:
                print(f"  {YELLOW}Open Source Models:{END}")
                for model in mv["llms"]:
                    print(f"    {CYAN}• {model}{END}")

                if "higher_parameter_model" in mv:
                    print(f"  {YELLOW}Higher Parameter Mappings:{END}")
                    for base_model, higher_model in mv["higher_parameter_model"].items():
                        print(f"    {CYAN}{base_model} → {higher_model}{END}")

            elif mv_mode == "commercial" and "commercial_model" in mv:
                print(f"  {YELLOW}Commercial Models:{END}")
                for model in mv["commercial_model"]:
                    print(f"    {CYAN}• {model}{END}")

        # RAG Configuration - Only display if method is RAG
        if "rag" in self.config and method_name == "RAG":
            print(f"\n{BOLD}{BLUE}🔍 RAG Configuration:{END}")
            rag = self.config["rag"]
            print(f"  {YELLOW}Embedding Model:{END} {GREEN}{rag.get('embedding_model', 'Not specified')}{END}")
            print(f"  {YELLOW}Chunking Strategy:{END} {GREEN}{rag.get('chunking_strategy', 'Not specified')}{END}")

            chunking_strategy = rag.get('chunking_strategy', '')
            if chunking_strategy == 'fixed_size' and 'chunk_size' in rag:
                print(f"  {YELLOW}Chunk Size:{END} {GREEN}{rag.get('chunk_size', 'Not specified')}{END}")
            elif chunking_strategy == 'small2big' and 'chunks_small2big' in rag:
                print(
                    f"  {YELLOW}Chunk Sizes (Small2Big):{END} {GREEN}{rag.get('chunks_small2big', 'Not specified')}{END}")
            elif chunking_strategy == 'sliding_window' and 'window_size' in rag:
                print(f"  {YELLOW}Window Size:{END} {GREEN}{rag.get('window_size', 'Not specified')}{END}")

            print(f"  {YELLOW}Similarity Cutoff:{END} {GREEN}{rag.get('similarity_cutoff', 'Not specified')}{END}")
            print(f"  {YELLOW}Top K:{END} {GREEN}{rag.get('top_k', 'Not specified')}{END}")
        elif "rag" in self.config and method_name != "RAG":
            print(f"\n{BOLD}{YELLOW}⚠️  RAG Configuration: {RED}Ignored (Method is not RAG){END}")

        # OpenAI Configuration
        if "OpenAI" in self.config:
            print(f"\n{BOLD}{BLUE}🔑 OpenAI Configuration:{END}")
            openai_config = self.config["OpenAI"]
            endpoint = openai_config.get("azure_endpoint", "")
            api_key = openai_config.get("api_key", "")
            print(f"  {YELLOW}Azure Endpoint:{END} {GREEN}{endpoint if endpoint else 'Not configured'}{END}")
            print(f"  {YELLOW}API Key:{END} {GREEN}{'Configured' if api_key else 'Not configured'}{END}")
            print(f"  {YELLOW}API Version:{END} {GREEN}{openai_config.get('api_version', 'Not specified')}{END}")

        # Evaluation Configuration
        if "evaluation" in self.config:
            print(f"\n{BOLD}{BLUE}📈 Evaluation Configuration:{END}")
            eval_config = self.config["evaluation"]
            if "metrics" in eval_config:
                metrics = eval_config["metrics"]
                print(f"  {YELLOW}Accuracy:{END} {GREEN}{metrics.get('accuracy', 'Not specified')}{END}")
                print(f"  {YELLOW}F1 Score:{END} {GREEN}{metrics.get('f1_score', 'Not specified')}{END}")

        # Other configurations
        if "openlit" in self.config:
            openlit_status = "Enabled" if self.config["openlit"] else "Disabled"
            color = GREEN if self.config["openlit"] else RED
            print(f"\n{BOLD}{BLUE}📊 OpenLit Monitoring:{END} {color}{openlit_status}{END}")

        if "output" in self.config:
            print(f"\n{BOLD}{BLUE}📁 Output Configuration:{END}")
            print(f"  {YELLOW}Directory:{END} {GREEN}{self.config['output'].get('directory', 'Not specified')}{END}")

        if "logging" in self.config:
            print(f"\n{BOLD}{BLUE}📝 Logging Configuration:{END}")
            logging_config = self.config["logging"]
            print(f"  {YELLOW}Level:{END} {GREEN}{logging_config.get('level', 'Not specified')}{END}")

        print("\n" + "=" * width)


def run_fact_check_experiment(config: Dict[str, Any]) -> None:
    """
    Main function to run the fact-check experiment with the given configuration.

    Args:
        config: Parsed configuration dictionary
    """
    print(f"\n{BOLD}{MAGENTA}🚀 Starting FactCheck Experiment...{END}")

    # Extract key parameters
    dataset_name = config.get("dataset", {}).get("name", "Unknown")
    method_name = config.get("method", {}).get("name", "Unknown")
    llm_config = config.get("llm", {})
    llm_model = llm_config.get("model", "Unknown")
    llm_mode = llm_config.get("mode", "Unknown")

    print(f"{CYAN}Dataset: {dataset_name}{END}")
    print(f"{CYAN}Method: {method_name}{END}")
    print(f"{CYAN}LLM: {llm_model} ({llm_mode}){END}")

    # Implement actual fact-checking logic
    print(f"{YELLOW}⏳ Processing...{END}")

    if method_name == "DKA":
        print(f"{GREEN}✓ Running DKA method...{END}")
        from methods.dka import run_dka_method
        run_dka_method(config)

    elif method_name == "GIV-Z":
        print(f"{GREEN}✓ Running GIV-Z method...{END}")
    elif method_name == "GIV-F":
        print(f"{GREEN}✓ Running GIV-F method...{END}")
    else:
        print(f"{GREEN}✓ Running RAG method...{END}")
        from methods.rag import run_rag_method
        run_rag_method(config)

    print(f"{GREEN}✅ Experiment completed successfully!{END}")


def main():
    """Main function to read configuration and run the experiment"""
    print(f"{BOLD}{CYAN}FactCheck Configuration Manager{END}")
    print("-" * 50)

    # Initialize config reader
    config_reader = ConfigReader("config.yml")

    # Load configuration
    config = config_reader.load_config()

    if not config:
        print(f"{RED}Failed to load configuration. Exiting.{END}")
        sys.exit(1)

    # Validate configuration
    validation_result = config_reader.validate_config()

    # Print validation results
    config_reader.print_validation_results()

    # Print configuration
    config_reader.print_configuration()

    # Only proceed if configuration is valid
    if validation_result.is_valid:
        # Run the experiment
        run_fact_check_experiment(config)
    else:
        print(f"\n{RED}❌ Configuration is invalid. Please fix the errors before proceeding.{END}")
        sys.exit(1)


if __name__ == "__main__":
    main()
