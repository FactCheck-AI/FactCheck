## 📋 Table of Contents

- [Overview](#overview)
- [DKA (Direct Knowledge Assessment) Prompts](#dka-direct-knowledge-assessment-prompts)
- [GIV (Guided Iterative Verification) Prompts](#giv-guided-iterative-verification-prompts)
- [RAG (Retrieval-Augmented Generation) Prompts](#rag-retrieval-augmented-generation-prompts)
- [Performance Impact](#performance-impact)
## 🎯 Overview

The FactCheck system employs multiple prompt strategies to evaluate the truthfulness of Knowledge Graph facts. Each methodology uses carefully crafted prompts optimized for different verification scenarios:

- **DKA**: Relies on LLM's internal knowledge without external guidance
- **GIV**: Provides structured guidelines and examples for verification
- **RAG**: Incorporates external evidence from web search results
- **Consensus**: Combines multiple model responses for improved accuracy

---

## 🧠 DKA (Direct Knowledge Assessment) Prompts

### Basic DKA Prompt (General Datasets)

```
Only reply with "T" or "F" and nothing else.
If the user provides a fact you judge accurate, reply with the letter "T".
If the user provides a fact you judge inaccurate, reply with the letter "F".

Fact: {{s}} {{p}} {{o}}

Judgment:
```

**Purpose:** Direct assessment using only LLM's internal knowledge
**Response Format:** Single character ("T" or "F")
**Use Cases:** FactBench, YAGO datasets

### DBpedia-Specific DKA Prompt

```
Your knowledge is limited to the year 2015. Only reply with "T" or "F" and nothing else.
If the user provides a fact you judge accurate (based on your knowledge up to 2015), reply with the
letter "T".
If the user provides a fact you judge inaccurate (based on your knowledge up to 2015), reply with the
letter "F".

Fact: {{s}} {{p}} {{o}}

Judgment:
```

**Purpose:** Time-constrained assessment for DBpedia dataset
**Knowledge Cutoff:** 2015 (matching DBpedia version)
**Response Format:** Single character ("T" or "F")

## 📋 GIV (Guided Iterative Verification) Prompts

### GIV-Z (Zero-Shot) Prompt Template

```
You are an expert fact-checker evaluating Knowledge Graph triples for accuracy.

TASK: Determine if the given fact is TRUE or FALSE.

INSTRUCTIONS:
- Respond with ONLY "T" for true facts or "F" for false facts
- Base your judgment on factual accuracy
- Consider the relationship between subject, predicate, and object
- If uncertain, lean towards the most likely answer based on your knowledge

FACT TO EVALUATE:
Subject: {{s}}
Predicate: {{p}} 
Object: {{o}}

RESPONSE (T or F):
```

**Purpose:** Structured verification with clear guidelines
**Response Format:** Single character ("T" or "F")
**Features:** Clear instructions, role definition, uncertainty handling

### GIV-F (Few-Shot) Prompt Template

```
You are an expert fact-checker evaluating Knowledge Graph triples for accuracy.

TASK: Determine if the given fact is TRUE or FALSE.

INSTRUCTIONS:
- Respond with ONLY "T" for true facts or "F" for false facts
- Base your judgment on factual accuracy
- Consider the relationship between subject, predicate, and object

EXAMPLES:
Example 1:
Subject: Barack Obama
Predicate: birthPlace
Object: Honolulu, Hawaii
Answer: T
(Barack Obama was indeed born in Honolulu, Hawaii)

Example 2:
Subject: Albert Einstein
Predicate: deathPlace
Object: London, England
Answer: F
(Albert Einstein died in Princeton, New Jersey, not London)

Example 3:
Subject: Microsoft
Predicate: foundedBy
Object: Bill Gates
Answer: T
(Bill Gates co-founded Microsoft)

NOW EVALUATE:
Subject: {{s}}
Predicate: {{p}}
Object: {{o}}

RESPONSE (T or F):
```

**Purpose:** Learning from examples to improve accuracy
**Response Format:** Single character ("T" or "F")
**Features:** Example-based learning, detailed explanations

## 🔍 RAG (Retrieval-Augmented Generation) Prompts

### Triple Transformation Prompt

```
Task Description:
Convert a knowledge graph triple into a meaningful human-readable sentence.

Instructions:
    Given a subject, predicate, and object from a knowledge graph, form a grammatically correct and meaningful sentence that conveys the relationship between them.

Examples:
Input:
    Subject: Alexander_III_of_Russia
    Predicate: isMarriedTo
    Object:  Maria_Feodorovna__Dagmar_of_Denmark_
    Output: {"output" : "Alexander III of Russia is married to Maria Feodorovna, also known as Dagmar of Denmark."}

Input: \
    Subject: Quentin_Tarantino
    Predicate: produced
    Object: From_Dusk_till_Dawn
    Output: {"output": "Quentin Tarantino produced the film From Dusk till Dawn."}

Input:
    Subject: Joseph_Heller
    Predicate: created
    Object: Catch-22
    Output: {"output": "Joseph Heller created the novel Catch-22."}

Do the following:
Input:
Subject: {s}
Predicate: {p}
Object: {o}
The output should be a JSON object with the key "output" and the value as the sentence. The sentence should be human-readable and grammatically correct. The subject, predicate, and object can be any valid string without having extra information.

```

**Purpose:** Convert structured triples to searchable text
**Output:** Human-readable sentence
**Use Case:** Preparing search queries

### Question Generation Prompt

```
You are an intelligent system with access to a vast amount of information.
I will provide you with a knowledge graph in the form of triples (subject, predicate, object).
Your task is to generate ten questions based on the knowledge graph. The questions should assess understanding and insight into the information presented in the graph.
Provide the output in JSON format, with each question having a unique identifier.
Instructions:
    1. Analyze the provided knowledge graph.
    2. Generate ten questions that are relevant to the information in the knowledge graph.
    3. Provide the questions in JSON format, each with a unique identifier.

Input Knowledge Graph:
    Albert Einstein bornIn Ulm
Expected Response:
{
    "questions": [
        {"id": 1, "question": "Where was Albert Einstein born?"},
        {"id": 2, "question": "What is Albert Einstein known for?"},
        {"id": 3, "question": "In what year was the Theory of Relativity published?"},
        {"id": 4, "question": "Where did Albert Einstein work?"},
        {"id": 5, "question": "What prestigious award did Albert Einstein win?"},
        {"id": 6, "question": "Which theory is associated with Albert Einstein?"},
        {"id": 7, "question": "Which university did Albert Einstein work at?"},
        {"id": 8, "question": "What did Albert Einstein receive the Nobel Prize in?"},
        {"id": 9, "question": "In what field did Albert Einstein win a Nobel Prize?"},
        {"id": 10, "question": "Name the city where Albert Einstein was born."}
]}


Considering the above information, please respond to this Knowledge Graph: {query}
The output should be in JSON format with each question having a unique identifier and question doesn't contain term knowledge graph, without any additional information 
"""
```

**Purpose:** Generate diverse verification questions
**Output:** 10 distinct questions
**Use Case:** Multi-angle fact exploration

### RAG Verification Prompt

```
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
Answer: 
```

**Purpose:** Evidence-based fact verification
**Input:** Fact + Retrieved evidence
**Response Format:** Single character ("T" or "F")

## 📊 Performance Impact

### Prompt Complexity vs. Accuracy
- **DKA:** Fast, relies on internal knowledge
- **GIV:** Better accuracy with guidelines/examples
- **RAG:** Highest accuracy with external evidence
- **Consensus:** Most robust through model combination

### Response Time Analysis
- **Simple Prompts (DKA):** ~0.3 seconds
- **Guided Prompts (GIV):** ~0.8 seconds
- **RAG Prompts:** ~2.3 seconds
- **Consensus:** ~1.5 seconds (parallel processing)

