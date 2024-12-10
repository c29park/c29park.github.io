## Modular RAG

![image](https://github.com/user-attachments/assets/8000facf-3754-4e68-85f0-c255db698a2f)
Modular RAG flow ([Source](https://github.com/langchain-ai/rag-from-scratch))

Basic RAG, often referred to as naive RAG, is a slightly outdated version. Not only is the pipeline too simple, but it also often struggles to understand complex user queries, retrieves irrelevant chunks, and ultimately result in hallucinated responses. Attempting to solve these issues, the state of the art version of RAG called Modular RAG incorporates various modules that can be modified and applied to enhance RAG performance. Langchain defines the modules as follows:
+ Query Translation 
+ Routing
+ Query Structuring
+ Indexing
+ Retrieval
+ Generation

----------------------------------
### Query Translation

![image](https://github.com/user-attachments/assets/84af8c55-25b1-47a7-80cb-0e28c5fbf13b)

The first module for modular RAG is Query Translation. This module transforms the query for better retreival of documents. Before we go through some techniques, we ought to know how a query can be transformed. When it comes to changing a certain statement or a question, we may either rewrite it in other words or in more general or specific terms while ensuring that it conveys the same meaning. In more technical terms, a query can be transformed by either using abstraction or paraphrasing. 

#### Multi-Query

![image](https://github.com/user-attachments/assets/26a2f8af-a0fd-4c49-80cb-a64390df4eea)

Multi-Query flow ([Source](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb))

Simply retrieving documents relevant to a user query makes the retrieval performance highly dependent on the quality of the user query. If a user query is complex and ambiguous, ambiguous documents would be retrieved as a result. To fix this problem, we can transform the query into multiple paraphrased questions such that at least one of the questions can be matched with some document(s) thereby improving document search. 

> [!Note]
> The code format for the pipeline depends on the framework on which the pipeline is based (e.g. Langchain, Haystack, LlamaIndex, etc). This post will use sample codes written by [Lance Martin](https://github.com/rlancemartin) from [Langchain's RAG from Scratch series](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) to explain the details.

Assuming that we use the same document for loading and the same method for indexing as those used in the previous post of basic RAG, here is a sample code of Multi-Query: 

```python
###PROMPT
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# Multi Query: Different Perspectives
template1 = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspectives 
    | ChatOpenAI(temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

### Retrieve
question = "What is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union

### RAG pipeline
template2 = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0)

final_rag_chain = (
    {"context": retrieval_chain, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"question":question})
```

As mentioned in the Multi-Query prompt `template1`, the LLM will provide multiple re-written questions separated by new lines given the original question from the user in the chain called `generate_queries`. The number of generated queries can be changed by writing an additional instruction in `template1`. Subsequently, `retrieval_chain` chains `generate_queries`, `retriever.map()`, and `get_unique_union`. As the image of the flow shows, the overall retrieval chain retrieves relevant documents to each generated query in parallel. This is why the `map()` function mapping the results of the retrieval was used on the `retriever`, which is essentially the vectorstore acting as a retriever. The retrieved documents for each query are formatted by the custom `get_unique_union` function, which returns a list of documents without any duplicates. For the final RAG chain, the `retrieval_chain` is passed as the overall context of the original question in the `template2`, and the rest is the same process seen in Basic RAG code. 

There is also an option for using the default `MultiQueryRetriever` provided by Langchain, of which the details can be seen [here](https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever). 
#### RAG-Fusion

![image](https://github.com/user-attachments/assets/db827943-0853-4fe4-b473-5e6b0557edf4)

RAG-Fusion flow ([Source](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb))

A slightly modified version of Multi-Query is RAG-Fusion, which adds an additional step of ranking the documents before giving the retrieved content as context to the LLM. The ranking process uses ranks the document based on a score called Reciprocal Rank Fusion (RRF), which is calculated as follows:

```math
RRFscore\left(d \in D \right) = \sum_{r \in R} \frac{1}{k + r\left(d\right)} 
```
where
- D is the set of all documents
- d is a document
- R is the set of rankers (retrievers)    
- k is a constant (typically 60)
- r(d) is the rank of document d in ranker r 

[Source](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) 

