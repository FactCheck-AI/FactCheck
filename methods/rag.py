"""
RAG (Retrieval-Augmented Generation) Method Implementation
"""

import time
import os
import logging
from typing import Dict, List, Any, Optional, cast

import json_repair
from llama_index.core import (
    Settings, VectorStoreIndex, SimpleDirectoryReader,
    PromptTemplate
)
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from pydantic import Field

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
query_engine = None
llm = None


class SimilarityNodePostprocessor(BaseNodePostprocessor):
    """Similarity-based Node processor for filtering retrieved documents."""

    knowledge_graph: str = Field(default="")
    similarity_cutoff: float = Field(default=0.3)

    @classmethod
    def class_name(cls) -> str:
        return "Similarity Cutoff Postprocessor"

    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes by filtering based on similarity cutoff."""
        new_nodes = []

        for node in nodes:
            should_use_node = True
            if self.similarity_cutoff is not None:
                similarity = node.score
                if similarity is None or cast(float, similarity) < cast(float, self.similarity_cutoff):
                    should_use_node = False

            if should_use_node:
                new_nodes.append(node)
            print(new_nodes)

        logger.info(f"Filtered {len(nodes)} nodes to {len(new_nodes)} nodes using similarity cutoff {self.similarity_cutoff}")
        return new_nodes

def few_shot_examples_fn(**kwargs):
    queries = [
        "Pat Frank author Alas, Babylon",
        "Elisabeth Domitien office Iraq",
        "Camilo José Cela award Nobel Prize in Literature",
    ]
    responses = [
        {"output": "yes"},
        {"output": "no"},
        {"output": "yes"},
    ]
    result_strs = []
    for query, response_dict in zip(queries, responses):
        result_str = f"""\
Query: {query}
Response: {response_dict}"""
        result_strs.append(result_str)
    return "\n\n".join(result_strs)

def get_directory_size(directory: str) -> int:
    """Calculate directory size in bytes."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
    except Exception as e:
        logger.warning(f"Could not calculate directory size: {e}")
    return total_size


def init_llm_settings(config: Dict[str, Any]) -> None:
    """Initialize LLM and embedding model settings for LlamaIndex."""
    global llm

    llm_config = config.get("llm", {})
    rag_config = config.get("rag", {})

    # Initialize LLM
    model_name = llm_config.get("model", "gemma2:9b")
    llm = Ollama(model=model_name, request_timeout=300.0)

    # Initialize embedding model
    embedding_model_name = rag_config.get("embedding_model", "BAAI/bge-small-en-v1.5")
    embed_model = HuggingFaceEmbedding(model_name=embedding_model_name, trust_remote_code=True)

    # Set global settings
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Configure node parser based on chunking strategy
    chunking_strategy = rag_config.get("chunking_strategy", "sliding_window")

    if chunking_strategy == "sliding_window":
        window_size = rag_config.get("window_size", 3)
        sentence_parser = SentenceWindowNodeParser.from_defaults(
            window_size=window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )
        Settings.node_parser = sentence_parser

    logger.info(f"Initialized LLM: {model_name}, Embedding: {embedding_model_name}")


def init_index(directory: str, collection_name: str, config: Dict[str, Any]) -> Optional[VectorStoreIndex]:
    """Initialize vector index from documents."""
    rag_config = config.get("rag", {})

    # Check directory size limit (70MB)
    if get_directory_size(directory) >= 70000000:
        logger.error(f"Directory {directory} size is greater than 70MB, can't create index")
        return None

    if not os.path.exists(directory):
        logger.warning(f"Directory {directory} does not exist")
        return None

    try:
        # Load documents
        reader = SimpleDirectoryReader(
            input_dir=directory,
            recursive=True,
            exclude=["questions.json", "index", "all_docs"]
        )
        documents = reader.load_data()

        if not documents:
            logger.warning(f"No documents found in {directory}")
            return None

        # Parse documents into nodes
        chunking_strategy = rag_config.get("chunking_strategy", "sliding_window")

        if chunking_strategy == "sliding_window":
            window_size = rag_config.get("window_size", 3)
            sentence_parser = SentenceWindowNodeParser.from_defaults(
                window_size=window_size,
                window_metadata_key="window",
                original_text_metadata_key="original_text"
            )
            sentence_nodes = sentence_parser.get_nodes_from_documents(documents)
        else:
            # Default to simple parsing
            sentence_nodes = Settings.node_parser.get_nodes_from_documents(documents)

        logger.info(f"Creating index with {len(documents)} documents, {len(sentence_nodes)} nodes")

        # Create vector index
        index = VectorStoreIndex(
            sentence_nodes,
            embed_model=Settings.embed_model,
            show_progress=False
        )

        return index

    except Exception as e:
        logger.error(f"Error creating index: {e}")
        return None


def create_formal_sentence(fact: Dict[str, Any]) -> str:
    # TODO: SHOULD BE FIXED SOMEHOW -- with the actual fact type and dataset wise
    """Convert fact to formal sentence."""
    s = fact.get("s", "")
    p = fact.get("p", "")
    o = fact.get("o", "")

    # Simple template for converting triple to natural language
    if p.lower() in ["born", "birthplace"]:
        return f"{s} was born in {o}"
    elif p.lower() in ["died", "deathplace"]:
        return f"{s} died in {o}"
    elif p.lower() in ["award", "prize"]:
        return f"{s} received the {o}"
    elif p.lower() in ["author", "wrote"]:
        return f"{s} authored {o}"
    else:
        return f"{s} {p} {o}"


def init_query_engine(index: VectorStoreIndex, query: str, config: Dict[str, Any]) -> None:
    """Initialize query engine with custom template and postprocessors."""
    global query_engine

    rag_config = config.get("rag", {})
    top_k = rag_config.get("top_k", 6)
    similarity_cutoff = rag_config.get("similarity_cutoff", 0.3)

    # RAG template for fact verification
    RAG_TEMPLATE = """
    Context information is below.
---------------------
{context_str}
---------------------
Given the context information and without prior knowledge, \
Evaluate whether the information in the documents supports the triple. \
Please provide your answer in the form of a structured JSON format containing \
a key \"output\" with the value as \"yes\" or \"no\". \
If the triple is correct according to the documents, the value should be \"yes\". \
If the triple is incorrect, the value should be \"no\". \

{few_shot_examples}

Query: {query_str}
Answer: """

    qa_template = PromptTemplate(RAG_TEMPLATE, function_mappings={"few_shot_examples": few_shot_examples_fn})

    # Configure postprocessors
    postprocessors = [
        SimilarityNodePostprocessor(
            knowledge_graph=query,
            similarity_cutoff=similarity_cutoff
        )
    ]

    # Add metadata replacement for sentence windows
    chunking_strategy = rag_config.get("chunking_strategy", "sliding_window")
    if chunking_strategy == "sliding_window":
        postprocessors.append(
            MetadataReplacementPostProcessor(target_metadata_key="window")
        )

    # Build query engine
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        text_qa_template=qa_template,
        node_postprocessors=postprocessors
    )

    logger.info(f"Initialized query engine with top_k={top_k}, similarity_cutoff={similarity_cutoff}")


def get_documents_directory(fact_id: str, dataset_name: str) -> str:
    """Get the documents directory for a specific fact."""
    # This should point to your RAG dataset directory structure
    # Adjust based on your actual directory structure
    base_dir = f"./rag_dataset/{dataset_name}/{fact_id}"
    return base_dir

def get_answer_from_llm(json_string: str, actual_answer: str) -> str:
    decoded_object = json_repair.repair_json(json_string, return_objects=True)
    if decoded_object['output'] == 'yes':
        return "T"
    elif decoded_object['output'] == 'no':
        return "F"
    else:
        # If the answer is not 'yes' or 'no', we can assume it's a mistake
        if actual_answer == "T":
            return "F"
        elif actual_answer == "F":
            return "T"


def run_rag_method(config: Dict[str, Any]) -> None:
    """
    Run RAG (Retrieval-Augmented Generation) method

    RAG augments the LLM with external knowledge through document retrieval
    """
    print(f"\n{BOLD}{BLUE}🔍 Running RAG (Retrieval-Augmented Generation) Method{END}")

    # Extract configuration
    dataset_name = config.get("dataset", {}).get("name", "Unknown")
    kg_ids = config.get("knowledge_graph", {}).get("kg_ids")
    llm_config = config.get("llm", {})
    rag_config = config.get("rag", {})

    print(f"{CYAN}Method: Retrieval-Augmented Generation{END}")
    print(f"{CYAN}Dataset: {dataset_name}{END}")
    print(f"{CYAN}LLM: {llm_config.get('model', 'Unknown')} ({llm_config.get('mode', 'Unknown')}){END}")
    print(f"{CYAN}Embedding Model: {rag_config.get('embedding_model', 'Unknown')}{END}")
    print(f"{CYAN}Chunking Strategy: {rag_config.get('chunking_strategy', 'Unknown')}{END}")

    # Load dataset
    print(f"\n{BOLD}📚 Loading Dataset...{END}")
    facts = load_dataset(dataset_name, kg_ids=kg_ids)

    if not facts:
        print(f"{RED}✗ No facts to process. Exiting RAG method.{END}")
        return

    # Initialize LLM settings for LlamaIndex
    print(f"\n{BOLD}🤖 Initializing LLM and Embedding Models...{END}")
    try:
        init_llm_settings(config)
        print(f"{GREEN}✓ LLM and embedding models initialized for RAG method{END}")
    except Exception as e:
        print(f"{RED}✗ Failed to initialize models: {str(e)}{END}")
        return

    # Initialize LLM client for final response generation
    try:
        llm_client = LLMClient(config)
        print(f"{GREEN}✓ LLM client initialized{END}")
    except Exception as e:
        print(f"{RED}✗ Failed to initialize LLM client: {str(e)}{END}")
        return

    # Process facts using RAG approach
    print(f"\n{BOLD}🔍 Processing Facts with RAG Method...{END}")
    print(f"{CYAN}RAG retrieves relevant documents to augment the model's knowledge{END}")

    results = []
    total_cost = 0.0
    total_tokens = 0
    successful_predictions = 0
    failed_retrievals = 0

    start_time = time.time()

    for i, fact_kg in enumerate(facts):
        identifier, fact = fact_kg
        print(f"\n{BLUE}Processing fact {i+1}/{len(facts)}: {identifier}{END}")
        print(f"  Fact: {fact.get('s', '')} {fact.get('p', '')} {fact.get('o', '')}")

        try:
            # Get documents directory for this fact
            docs_directory = get_documents_directory(identifier, dataset_name)

            if not os.path.exists(docs_directory):
                print(f"  {YELLOW}⚠️ No documents found for {identifier}, skipping...{END}")
                failed_retrievals += 1

                # Create failed result
                result = {
                    "id": identifier,
                    "method": "RAG",
                    "fact": {
                        "s": fact.get("s", ""),
                        "p": fact.get("p", ""),
                        "o": fact.get("o", ""),
                        "label": fact.get("label")
                    },
                    "prompt": "No documents available for retrieval",
                    "response": 'F' if fact.get("label") == 'T' else 'T',
                    "success": False,
                    "error_message": f"No documents directory found: {docs_directory}",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                results.append(result)
                continue

            # Create index for this fact's documents
            print(f"  {CYAN}Creating index from documents...{END}")
            collection_name = f"{identifier}_collection"
            index = init_index(docs_directory, collection_name, config)

            if index is None:
                print(f"  {RED}✗ Failed to create index for {identifier}{END}")
                failed_retrievals += 1

                result = {
                    "id": identifier,
                    "method": "RAG",
                    "fact": {
                        "s": fact.get("s", ""),
                        "p": fact.get("p", ""),
                        "o": fact.get("o", ""),
                        "label": fact.get("label")
                    },
                    "prompt": "Failed to create document index",
                    "response": "F",
                    "success": False,
                    "error_message": "Failed to create document index",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                results.append(result)
                continue

            # Convert fact to natural language query
            formal_sentence = create_formal_sentence(fact)
            print(f"  {CYAN}Query: {formal_sentence}{END}")

            # Initialize query engine
            init_query_engine(index, formal_sentence, config)

            # Retrieve relevant context
            print(f"  {CYAN}Retrieving relevant documents...{END}")
            retrieval_start = time.time()

            try:
                # Query the index to get relevant context
                response = query_engine.query(formal_sentence)
                answer = response.response if hasattr(response, 'response') else str(response)
                final_answer = get_answer_from_llm(answer, fact.get("label", "F"))
                retrieval_time = time.time() - retrieval_start

                print(f"  {GREEN}✓ Retrieved context in {retrieval_time:.2f}s{END}")

                # Process result
                result = {
                    "id": identifier,
                    "method": "RAG",
                    "fact": {
                        "s": fact.get("s", ""),
                        "p": fact.get("p", ""),
                        "o": fact.get("o", ""),
                        "label": fact.get("label")
                    },
                    # "prompt": "rag_prompt",
                    "full_answer": answer,
                    "response": final_answer,
                    "success": hasattr(response, 'response'),
                    # "error_message": 'llm_response.error_message',
                    "response_time": retrieval_time,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                if hasattr(response, 'response'):
                    successful_predictions += 1
                    predicted = final_answer.strip().upper()
                    print(f"  {GREEN}RAG Prediction: {predicted}{END}")

                    # Show accuracy if ground truth is available
                    if fact.get("label") is not None:
                        actual = "T" if fact["label"] else "F"
                        correct = "✓" if predicted == actual else "✗"
                        color = GREEN if predicted == actual else RED
                        print(f"  {color}Ground Truth: {actual} {correct}{END}")
                else:
                    print(f"  {RED}Failed: error occurred{END}")

            except Exception as e:
                print(f"  {RED}✗ Error during RAG query: {str(e)}{END}")
                result = {
                    "id": identifier,
                    "method": "RAG",
                    "fact": {
                        "s": fact.get("s", ""),
                        "p": fact.get("p", ""),
                        "o": fact.get("o", ""),
                        "label": fact.get("label")
                    },
                    "prompt": f"Error during RAG processing: {str(e)}",
                    "response": "F",
                    "success": False,
                    "error_message": f"RAG processing error: {str(e)}",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }

        except Exception as e:
            print(f"  {RED}✗ Unexpected error processing {identifier}: {str(e)}{END}")
            result = {
                "id": identifier,
                "method": "RAG",
                "fact": {
                    "s": fact.get("s", ""),
                    "p": fact.get("p", ""),
                    "o": fact.get("o", ""),
                    "label": fact.get("label")
                },
                "prompt": f"Unexpected error: {str(e)}",
                "response": "F",
                "success": False,
                "error_message": f"Unexpected error: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

        results.append(result)

    # Calculate final metrics
    end_time = time.time()
    duration = end_time - start_time

    # Print RAG summary
    print(f"\n{BOLD}{GREEN}📊 RAG Method Summary{END}")
    print(f"  Method: Retrieval-Augmented Generation")
    print(f"  Total facts processed: {len(facts)}")
    print(f"  Successful predictions: {successful_predictions}")
    print(f"  Failed predictions: {len(facts) - successful_predictions}")
    print(f"  Failed retrievals: {failed_retrievals}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Avg time per fact: {duration/len(facts):.2f} seconds")

    # Save results
    print(f"\n{BOLD}💾 Saving RAG Results...{END}")
    results_file = save_results('rag', results, config)

    if results_file:
        print(f"{GREEN}✅ RAG method completed successfully!{END}")
        print(f"Results saved to: {results_file}")
    else:
        print(f"{YELLOW}⚠️ RAG method completed but failed to save results{END}")


if __name__ == "__main__":
    # Example usage for testing
    sample_config = {
        "dataset": {"name": "FactBench"},
        "method": {"name": "RAG"},
        "llm": {
            "mode": "open_source",
            "model": "gemma2:9b",
            "parameters": {"temperature": 0.7}
        },
        "rag": {
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "chunking_strategy": "sliding_window",
            "window_size": 3,
            "similarity_cutoff": 0.3,
            "top_k": 6
        },
        "knowledge_graph": {"kg_ids": ['correct_death_00106']},
        "output": {"directory": "./results"}
    }

    run_rag_method(sample_config)