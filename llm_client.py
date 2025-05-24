import requests
import json
import time
import random
import openai
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import logging

from datetime import datetime, timedelta

from openai import AzureOpenAI

# Define colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
END = "\033[0m"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Response from LLM request"""
    content: str
    success: bool
    error_message: Optional[str] = None
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    # cost: Optional[float] = None
    response_time: Optional[float] = None
    model_used: Optional[str] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

# class RateLimiter:
#     """Simple rate limiter for API calls"""
#
#     def __init__(self, max_calls_per_minute: int = 60):
#         self.max_calls = max_calls_per_minute
#         self.calls = []
#         self.lock = threading.Lock()
#
#     def wait_if_needed(self):
#         """Wait if rate limit would be exceeded"""
#         with self.lock:
#             now = datetime.now()
#             # Remove calls older than 1 minute
#             self.calls = [call_time for call_time in self.calls
#                           if now - call_time < timedelta(minutes=1)]
#
#             if len(self.calls) >= self.max_calls:
#                 # Wait until the oldest call is more than 1 minute old
#                 sleep_time = 60 - (now - self.calls[0]).total_seconds()
#                 if sleep_time > 0:
#                     logger.info(f"Rate limit reached, waiting {sleep_time:.1f} seconds...")
#                     time.sleep(sleep_time)
#
#             self.calls.append(now)
#
# class TokenCounter:
#     """Utility class for counting tokens and estimating costs"""
#
#     # Rough token counting (for more accuracy, use tiktoken library)
#     @staticmethod
#     def estimate_tokens(text: str) -> int:
#         """Rough estimation of token count"""
#         # Very rough approximation: ~4 characters per token for English
#         return len(text) // 4 + len(text.split()) // 2
#
#     @staticmethod
#     def calculate_cost(tokens: int, model: str) -> float:
#         """Calculate cost based on tokens and model"""
#         # Updated pricing as of 2024 (adjust based on current pricing)
#         cost_per_1k_tokens = {
#             # OpenAI Models
#             "gpt-4o": 0.005,           # Input tokens
#             "gpt-4o-mini": 0.00015,    # Input tokens
#             "gpt-4": 0.03,             # Input tokens
#             "gpt-4-turbo": 0.01,       # Input tokens
#             "gpt-3.5-turbo": 0.0005,   # Input tokens
#
#             # Azure OpenAI (same pricing structure)
#             "gpt-35-turbo": 0.0005,
#             "gpt-4-32k": 0.06,
#
#             # DeepSeek Models (example pricing)
#             "deepseek-v3": 0.002,
#             "deepseek-r1": 0.0015,
#         }
#
#         model_lower = model.lower()
#         rate = cost_per_1k_tokens.get(model_lower, 0.001)  # Default rate
#         return (tokens / 1000) * rate

class LLMClient:
    """Comprehensive LLM client supporting both commercial and open source models"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config.get("llm", {})
        self.openai_config = config.get("OpenAI", {})
        self.mode = self.llm_config.get("mode", "open_source")
        self.model = self.llm_config.get("model", "")
        self.parameters = self.llm_config.get("parameters", {})
        self.azure_client = None

        # Rate limiting
        # self.rate_limiter = RateLimiter(max_calls_per_minute=60)

        # Initialize clients based on mode
        self.client_initialized = False
        self._init_client()

    def _init_client(self):
        """Initialize the appropriate client based on mode"""
        try:
            if self.mode == "commercial":
                self._init_commercial_client()
            else:
                self._init_ollama_client()
            self.client_initialized = True
            logger.info(f"✓ LLM client initialized for {self.mode} mode with model: {self.model}")
        except Exception as e:
            logger.error(f"✗ Failed to initialize LLM client: {str(e)}")
            raise

    def _init_commercial_client(self):
        """Initialize commercial (OpenAI/Azure) client"""
        if not self.openai_config.get("api_key"):
            raise ValueError("OpenAI API key is required for commercial models")

        if not self.openai_config.get("azure_endpoint"):
            raise ValueError("Azure endpoint is required for Azure OpenAI models")


        self.azure_client = AzureOpenAI(
            azure_endpoint = self.openai_config.get("azure_endpoint"),
            api_key= self.openai_config.get("api_key"),
            api_version= self.openai_config.get("api_version", "2024-12-01-preview")
        )


    def _init_ollama_client(self):
        """Initialize Ollama client"""
        self.ollama_url = "http://localhost:11434"
        self.ollama_generate_url = f"{self.ollama_url}/api/generate"
        self.ollama_chat_url = f"{self.ollama_url}/api/chat"

        # Check if Ollama is running
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                available_models = response.json().get("models", [])
                model_names = [model["name"] for model in available_models]
                logger.info(f"Ollama is running with {len(model_names)} models available")

                # Check if our model is available
                if self.model.lower() not in [m.lower() for m in model_names]:
                    logger.warning(f"Model '{self.model}' not found in Ollama. Available: {model_names}")
                    # Try to pull the model
                    self._pull_ollama_model(self.model)
            else:
                raise ConnectionError(f"Ollama returned status code: {response.status_code}")

        except requests.RequestException as e:
            raise ConnectionError(f"Ollama is not running or not accessible: {str(e)}. "
                                  f"Please start it with 'ollama serve'")

    def _pull_ollama_model(self, model_name: str) -> bool:
        """Attempt to pull a model in Ollama"""
        try:
            logger.info(f"Attempting to pull model: {model_name}")
            pull_url = f"{self.ollama_url}/api/pull"
            response = requests.post(pull_url, json={"name": model_name}, timeout=300)

            if response.status_code == 200:
                logger.info(f"✓ Successfully pulled model: {model_name}")
                return True
            else:
                logger.error(f"✗ Failed to pull model {model_name}: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"✗ Error pulling model {model_name}: {str(e)}")
            return False

    def generate_response(self, prompt: str, max_retries: int = 3,
                          timeout: int = 60) -> LLMResponse:
        """
        Generate response from LLM with retries and comprehensive error handling

        Args:
            prompt: Input prompt for the LLM
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds

        Returns:
            LLMResponse object containing the result
        """
        if not self.client_initialized:
            return LLMResponse(
                content="",
                success=False,
                error_message="LLM client not properly initialized",
                timestamp=datetime.now().isoformat()
            )

        start_time = time.time()
        last_error = None

        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                # self.rate_limiter.wait_if_needed()

                # Add exponential backoff for retries
                if attempt > 0:
                    backoff_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {backoff_time:.1f} seconds (attempt {attempt + 1}/{max_retries})")
                    time.sleep(backoff_time)

                # Make the API call
                if self.mode == "commercial":
                    response = self._call_commercial_llm(prompt, timeout)
                else:
                    response = self._call_ollama_llm(prompt, timeout)

                # Add timing information
                response.response_time = time.time() - start_time
                response.model_used = self.model
                response.timestamp = datetime.now().isoformat()

                if response.success:
                    logger.debug(f"✓ Successful response from {self.model} in {response.response_time:.2f}s")
                    return response
                else:
                    last_error = response.error_message
                    logger.warning(f"✗ Attempt {attempt + 1} failed: {response.error_message}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"✗ Attempt {attempt + 1} failed with exception: {str(e)}")

                # Don't retry on certain types of errors
                if any(error_type in str(e).lower() for error_type in
                       ["invalid_api_key", "unauthorized", "forbidden", "model_not_found"]):
                    break

        # All retries failed
        total_time = time.time() - start_time
        return LLMResponse(
            content="",
            success=False,
            error_message=f"All {max_retries} attempts failed. Last error: {last_error}",
            response_time=total_time,
            model_used=self.model,
            timestamp=datetime.now().isoformat()
        )

    def _call_commercial_llm(self, prompt: str, timeout: int) -> LLMResponse:
        """Call commercial LLM (OpenAI/Azure)"""
        try:
            # Prepare the request parameters
            request_params = {
                "model": self.model,
                "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": prompt}],
                "temperature": self.parameters.get("temperature", 0.7),
                "top_p": self.parameters.get("top_p", 0.9),
                "max_tokens": self.parameters.get("max_tokens", 512),
                "timeout": timeout
            }

            # Make the API call
            response = self.azure_client.chat.completions.create(
                messages=[{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": prompt}],
                model=self.model,
            )

            # Extract response data
            content = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

            # Calculate cost
            # cost = self._calculate_commercial_cost(prompt_tokens, completion_tokens, self.model)

            # logger.debug(f"Response: {len(content)} chars, {tokens_used} tokens, ${cost:.6f}")
            logger.debug(f"Response: {len(content)} chars, {tokens_used}")

            return LLMResponse(
                content=content,
                success=True,
                tokens_used=tokens_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                # cost=cost
            )
        except Exception as e:
            return LLMResponse(
                content="",
                success=False,
                error_message=f"Commercial LLM error: {str(e)}"
            )

    def _call_ollama_llm(self, prompt: str, timeout: int) -> LLMResponse:
        """Call Ollama LLM"""
        try:
            # Prepare the request payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.parameters.get("temperature", 0.7),
                    "top_p": self.parameters.get("top_p", 0.9),
                    "top_k": self.parameters.get("top_k", 40),
                    "num_predict": self.parameters.get("max_tokens", 512),
                    "repeat_penalty": self.parameters.get("repeat_penalty", 1.1),
                    "stop": self.parameters.get("stop", [])
                }
            }

            logger.debug(f"Making Ollama request to {self.model}")

            # Make the API call
            response = requests.post(
                self.ollama_generate_url,
                json=payload,
                timeout=timeout,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code != 200:
                error_msg = f"Ollama API returned status {response.status_code}"
                try:
                    error_detail = response.json().get("error", "Unknown error")
                    error_msg += f": {error_detail}"
                except:
                    error_msg += f": {response.text[:200]}"

                return LLMResponse(
                    content="",
                    success=False,
                    error_message=error_msg
                )

            # Parse the response
            result = response.json()
            content = result.get("response", "").strip()

            # Ollama doesn't provide token counts by default, so estimate
            # estimated_tokens = TokenCounter.estimate_tokens(prompt + content)
            estimated_tokens = 2
            logger.debug(f"Ollama response: {len(content)} chars, ~{estimated_tokens} tokens")

            return LLMResponse(
                content=content,
                success=True,
                # tokens_used=estimated_tokens,
                # cost=0.0  # Open source models are free
            )

        except requests.Timeout:
            return LLMResponse(
                content="",
                success=False,
                error_message=f"Request timeout after {timeout} seconds"
            )
        except requests.ConnectionError:
            return LLMResponse(
                content="",
                success=False,
                error_message="Connection error: Is Ollama running?"
            )
        except json.JSONDecodeError as e:
            return LLMResponse(
                content="",
                success=False,
                error_message=f"Invalid JSON response from Ollama: {str(e)}"
            )
        except Exception as e:
            return LLMResponse(
                content="",
                success=False,
                error_message=f"Ollama LLM error: {str(e)}"
            )

    # def _calculate_commercial_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
    #     """Calculate cost for commercial models with input/output pricing"""
    #     # Updated pricing structure (input/output tokens priced differently)
    #     pricing = {
    #         "gpt-4o": {"input": 0.005, "output": 0.015},
    #         "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    #         "gpt-4": {"input": 0.03, "output": 0.06},
    #         "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    #         "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    #         "gpt-35-turbo": {"input": 0.0005, "output": 0.0015},  # Azure naming
    #     }
    #
    #     model_key = model.lower()
    #     if model_key in pricing:
    #         input_cost = (prompt_tokens / 1000) * pricing[model_key]["input"]
    #         output_cost = (completion_tokens / 1000) * pricing[model_key]["output"]
    #         return input_cost + output_cost
    #     else:
    #         # Fallback to simple calculation
    #         return TokenCounter.calculate_cost(prompt_tokens + completion_tokens, model)

    # def batch_generate(self, prompts: List[str], show_progress: bool = True,
    #                    max_workers: int = 3) -> List[LLMResponse]:
    #     """
    #     Generate responses for multiple prompts with optional parallel processing
    #
    #     Args:
    #         prompts: List of prompts to process
    #         show_progress: Whether to show progress bar
    #         max_workers: Maximum number of parallel workers
    #
    #     Returns:
    #         List of LLMResponse objects
    #     """
    #     responses = []
    #     total_prompts = len(prompts)
    #
    #     if show_progress:
    #         print(f"{BLUE}Processing {total_prompts} prompts...{END}")
    #
    #     # For commercial APIs, use conservative parallelism to avoid rate limits
    #     if self.mode == "commercial":
    #         max_workers = min(max_workers, 2)
    #
    #     if max_workers > 1:
    #         # Parallel processing
    #         with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #             # Submit all tasks
    #             future_to_prompt = {
    #                 executor.submit(self.generate_response, prompt): i
    #                 for i, prompt in enumerate(prompts)
    #             }
    #
    #             # Collect results as they complete
    #             results = [None] * total_prompts
    #             completed = 0
    #
    #             for future in as_completed(future_to_prompt):
    #                 prompt_idx = future_to_prompt[future]
    #                 try:
    #                     response = future.result()
    #                     results[prompt_idx] = response
    #                 except Exception as e:
    #                     results[prompt_idx] = LLMResponse(
    #                         content="",
    #                         success=False,
    #                         error_message=f"Parallel processing error: {str(e)}"
    #                     )
    #
    #                 completed += 1
    #                 if show_progress:
    #                     progress = completed / total_prompts * 100
    #                     print(f"\r{BLUE}Progress: {progress:.1f}% ({completed}/{total_prompts}){END}",
    #                           end="", flush=True)
    #
    #             responses = results
    #     else:
    #         # Sequential processing
    #         for i, prompt in enumerate(prompts):
    #             if show_progress:
    #                 progress = (i + 1) / total_prompts * 100
    #                 print(f"\r{BLUE}Progress: {progress:.1f}% ({i + 1}/{total_prompts}){END}",
    #                       end="", flush=True)
    #
    #             response = self.generate_response(prompt)
    #             responses.append(response)
    #
    #             # Add delay for rate limiting
    #             if self.mode == "commercial":
    #                 time.sleep(0.1)
    #
    #     if show_progress:
    #         print()  # New line after progress bar
    #
    #         # Print summary
    #         successful = sum(1 for r in responses if r.success)
    #         failed = total_prompts - successful
    #         total_cost = sum(r.cost or 0 for r in responses)
    #         total_tokens = sum(r.tokens_used or 0 for r in responses)
    #
    #         print(f"{GREEN}✓ Batch completed: {successful} successful, {failed} failed{END}")
    #         print(f"{CYAN}Total tokens: {total_tokens}, Total cost: ${total_cost:.4f}{END}")
    #
    #     return responses

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        info = {
            "model": self.model,
            "mode": self.mode,
            "parameters": self.parameters,
            "client_initialized": self.client_initialized
        }

        if self.mode == "commercial":
            info["api_type"] = getattr(openai, "api_type", "unknown")
            info["api_base"] = getattr(openai, "api_base", "unknown")
        else:
            info["ollama_url"] = self.ollama_url

        return info

# class MajorityVoteLLMClient:
#     """Extended LLM client for majority voting across multiple models"""
#
#     def __init__(self, config: Dict[str, Any]):
#         self.config = config
#         self.mv_config = config.get("majority_vote", {})
#         self.mode = self.mv_config.get("mode", "open_source")
#         self.num_votes = self.mv_config.get("num_votes", 3)
#
#         # Initialize multiple clients for voting
#         self.clients: List[LLMClient] = []
#         self.model_names: List[str] = []
#         self._init_voting_clients()
#
#     def _init_voting_clients(self):
#         """Initialize multiple LLM clients for voting"""
#         if self.mode == "commercial":
#             commercial_models = self.mv_config.get("commercial_model", [])
#             if not commercial_models:
#                 raise ValueError("commercial_model list is required for commercial majority voting")
#
#             for model in commercial_models[:self.num_votes]:
#                 client_config = self.config.copy()
#                 client_config["llm"]["model"] = model
#                 client_config["llm"]["mode"] = "commercial"
#
#                 try:
#                     client = LLMClient(client_config)
#                     self.clients.append(client)
#                     self.model_names.append(model)
#                     logger.info(f"✓ Initialized commercial client for {model}")
#                 except Exception as e:
#                     logger.error(f"✗ Failed to initialize client for {model}: {str(e)}")
#
#         else:  # open_source
#             llms = self.mv_config.get("llms", [])
#             higher_param_models = self.mv_config.get("higher_parameter_model", {})
#
#             if not llms:
#                 raise ValueError("llms list is required for open source majority voting")
#
#             for model in llms[:self.num_votes]:
#                 # Use higher parameter model if available
#                 model_to_use = higher_param_models.get(model.lower(), model)
#
#                 client_config = self.config.copy()
#                 client_config["llm"]["model"] = model_to_use
#                 client_config["llm"]["mode"] = "open_source"
#
#                 try:
#                     client = LLMClient(client_config)
#                     self.clients.append(client)
#                     self.model_names.append(model_to_use)
#                     logger.info(f"✓ Initialized open source client for {model_to_use}")
#                 except Exception as e:
#                     logger.error(f"✗ Failed to initialize client for {model_to_use}: {str(e)}")
#
#         if not self.clients:
#             raise ValueError("No LLM clients could be initialized for majority voting")
#
#         logger.info(f"✓ Majority voting initialized with {len(self.clients)} models")
#
#     def majority_vote_generate(self, prompt: str, timeout: int = 60,
#                                parallel: bool = True) -> LLMResponse:
#         """
#         Generate response using majority voting across multiple models
#
#         Args:
#             prompt: Input prompt for the LLMs
#             timeout: Request timeout for each model
#             parallel: Whether to run models in parallel
#
#         Returns:
#             LLMResponse with majority vote result
#         """
#         print(f"{BLUE}🗳️  Running majority vote with {len(self.clients)} models...{END}")
#
#         responses = []
#         votes = {}
#         total_cost = 0.0
#         total_tokens = 0
#
#         start_time = time.time()
#
#         if parallel and len(self.clients) > 1:
#             # Parallel execution
#             with ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
#                 future_to_client = {
#                     executor.submit(client.generate_response, prompt, timeout=timeout): (i, client)
#                     for i, client in enumerate(self.clients)
#                 }
#
#                 for future in as_completed(future_to_client):
#                     client_idx, client = future_to_client[future]
#                     model_name = self.model_names[client_idx]
#
#                     try:
#                         response = future.result()
#                         responses.append((model_name, response))
#
#                         if response.success:
#                             # Update totals
#                             if response.tokens_used:
#                                 total_tokens += response.tokens_used
#                             if response.cost:
#                                 total_cost += response.cost
#
#                             # Count vote
#                             vote = self._normalize_response(response.content)
#                             votes[vote] = votes.get(vote, 0) + 1
#                             print(f"  {CYAN}{model_name}:{END} {response.content} → {vote}")
#                         else:
#                             print(f"  {RED}{model_name} failed:{END} {response.error_message}")
#
#                     except Exception as e:
#                         print(f"  {RED}{model_name} error:{END} {str(e)}")
#         else:
#             # Sequential execution
#             for i, client in enumerate(self.clients):
#                 model_name = self.model_names[i]
#                 print(f"  {YELLOW}Querying {model_name}...{END}")
#
#                 response = client.generate_response(prompt, timeout=timeout)
#                 responses.append((model_name, response))
#
#                 if response.success:
#                     # Update totals
#                     if response.tokens_used:
#                         total_tokens += response.tokens_used
#                     if response.cost:
#                         total_cost += response.cost
#
#                     # Count vote
#                     vote = self._normalize_response(response.content)
#                     votes[vote] = votes.get(vote, 0) + 1
#                     print(f"  {GREEN}{model_name}:{END} {response.content} → {vote}")
#                 else:
#                     print(f"  {RED}{model_name} failed:{END} {response.error_message}")
#
#         # Determine majority vote
#         if not votes:
#             return LLMResponse(
#                 content="",
#                 success=False,
#                 error_message="No successful responses from any model",
#                 tokens_used=total_tokens,
#                 cost=total_cost,
#                 response_time=time.time() - start_time,
#                 model_used=f"majority_vote({len(self.clients)}_models)",
#                 timestamp=datetime.now().isoformat()
#             )
#
#         # Find majority vote
#         majority_vote = max(votes, key=votes.get)
#         confidence = votes[majority_vote] / sum(votes.values())
#         successful_responses = sum(1 for _, resp in responses if resp.success)
#
#         print(f"  {GREEN}📊 Vote Results:{END}")
#         for vote, count in sorted(votes.items(), key=lambda x: x[1], reverse=True):
#             percentage = count / sum(votes.values()) * 100
#             print(f"    {vote}: {count}/{sum(votes.values())} ({percentage:.1f}%)")
#
#         print(f"  {BOLD}{GREEN}🏆 Majority Vote: {majority_vote} (confidence: {confidence:.2%}){END}")
#
#         # Create detailed response
#         response_details = {
#             "individual_responses": [
#                 {"model": model, "response": resp.content, "success": resp.success,
#                  "error": resp.error_message}
#                 for model, resp in responses
#             ],
#             "vote_distribution": votes,
#             "confidence": confidence,
#             "successful_models": successful_responses,
#             "total_models": len(self.clients)
#         }
#
#         return LLMResponse(
#             content=majority_vote,
#             success=True,
#             tokens_used=total_tokens,
#             cost=total_cost,
#             response_time=time.time() - start_time,
#             model_used=f"majority_vote({successful_responses}/{len(self.clients)}_models)",
#             timestamp=datetime.now().isoformat(),
#             error_message=json.dumps(response_details)  # Store details in error_message field
#         )
#
#     def _normalize_response(self, response: str) -> str:
#         """Normalize response to standard format (T/F)"""
#         response = response.strip().upper()
#
#         # Direct matches
#         if response in ['T', 'F', 'TRUE', 'FALSE']:
#             return 'T' if response.startswith('T') else 'F'
#
#         # Pattern matching
#         if any(word in response for word in ['TRUE', 'CORRECT', 'YES', 'ACCURATE']):
#             return 'T'
#         elif any(word in response for word in ['FALSE', 'INCORRECT', 'NO', 'INACCURATE']):
#             return 'F'
#         elif response.startswith('T'):
#             return 'T'
#         elif response.startswith('F'):
#             return 'F'
#         else:
#             # If unclear, return the first character or original response
#             return response[0] if response else 'UNCLEAR'
#
#     def get_models_info(self) -> List[Dict[str, Any]]:
#         """Get information about all models in the voting ensemble"""
#         return [
#             {
#                 "model_name": name,
#                 "model_info": client.get_model_info()
#             }
#             for name, client in zip(self.model_names, self.clients)
#         ]

# Utility functions for external use
def create_llm_client(config: Dict[str, Any]) -> LLMClient:
    """
    Factory function to create the appropriate LLM client

    Args:
        config: Configuration dictionary

    Returns:
        LLMClient based on configuration
    """
    return LLMClient(config)

def test_llm_client(config: Dict[str, Any]) -> None:
    """
    Test function to verify LLM client functionality

    Args:
        config: Configuration dictionary
    """
    print(f"{BOLD}{BLUE}🧪 Testing LLM Client{END}")

    try:
        # Create client
        client = create_llm_client(config)

        # Test prompt
        test_prompt = "Reply with just the letter 'T' if Paris is the capital of France, or 'F' if not."

        print(f"{YELLOW}Test prompt: {test_prompt}{END}")

        # Make request
        if hasattr(client, 'majority_vote_generate'):
            response = client.majority_vote_generate(test_prompt)
        else:
            response = client.generate_response(test_prompt)

        # Display results
        if response.success:
            print(f"{GREEN}✓ Test successful!{END}")
            print(f"  Response: {response.content}")
            print(f"  Tokens: {response.tokens_used}")
            print(f"  Cost: ${response.cost:.6f}")
            print(f"  Time: {response.response_time:.2f}s")
        else:
            print(f"{RED}✗ Test failed: {response.error_message}{END}")

    except Exception as e:
        print(f"{RED}✗ Client test failed: {str(e)}{END}")

if __name__ == "__main__":
    # Example usage
    sample_config = {
        "llm": {
            "mode": "open_source",
            "model": "qwen2.5:7b",
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 512
            }
        },
        "OpenAI": {
            "api_key": "",
            "azure_endpoint": "",
            "api_version": "2024-02-15-preview"
        }
    }

    test_llm_client(sample_config)