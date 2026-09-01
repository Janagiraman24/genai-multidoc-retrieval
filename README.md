## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
Extracting specific, nuanced information from a collection of dense academic papers is a slow and inefficient manual process. Standard search tools rely on exact keywords and fail to understand the conceptual context of a user's question. This program aims to build an AI agent that can intelligently query multiple documents to synthesize precise answers to complex questions.

### DESIGN STEPS:

#### STEP 1: Load PDF documents and create specialized search and summary tools for each paper.

#### STEP 2: Initialize an AI agent with an OpenAI model, giving it access to all the created tools.

#### STEP 3:  Query the agent with a specific question about one paper to get a detailed answer from its content.

### PROGRAM:
```python
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
```
```python
import nest_asyncio
nest_asyncio.apply()
```
```python
urls = [
    "https://openreview.net/pdf?id=jHDZEUgS4r",
    "https://openreview.net/pdf?id=CCSPm6V5EF",
    "https://openreview.net/pdf?id=krGpQzo8Mz"
    
]

papers = [
    "MedAgentGym.pdf",
    "WebDevJudge.pdf",
    "Latent_Speech.pdf"
]
```
```python
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
```
```python
initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
```
```python
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
```
```python
len(initial_tools)
```
```python
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)
```
```python
print("Name: JANAGIRAMAN M")
print("Register Number: 212224230101")
response = agent.query(
    "What is MedAgentGym?"
    "Why was WEBDEVJUDGE introduced?"
    "Why are speech-text models computationally expensive?"
)
```
```python
response = agent.query("Give me a summary of all the 3 documents")
print(str(response))
```

### OUTPUT:
<img width="617" height="85" alt="image" src="https://github.com/user-attachments/assets/7620df29-2ed5-4db3-9a54-681627dc815d" />

<img width="488" height="42" alt="image" src="https://github.com/user-attachments/assets/59db33de-d68a-497b-8e23-df7906825145" />

<img width="1156" height="772" alt="image" src="https://github.com/user-attachments/assets/fab9402c-0a1e-4770-a051-d0f1b90c19c5" />
<img width="1180" height="130" alt="image" src="https://github.com/user-attachments/assets/615487de-ad3d-4f03-904c-ddcc241b2a97" />

<img width="1200" height="737" alt="image" src="https://github.com/user-attachments/assets/e8e6bdb5-3ad6-4dff-895b-35570a327919" />
<img width="1206" height="667" alt="image" src="https://github.com/user-attachments/assets/54696456-2b14-4248-b865-cdb45717b7f7" />



### RESULT:
The system successfully retrieves and synthesizes relevant information from multiple documents, providing concise and relevant answers to the user's query. Performance is evaluated based on the accuracy, relevance, and coherence of the responses.
