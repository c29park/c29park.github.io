## Modular RAG

![image](https://github.com/user-attachments/assets/8000facf-3754-4e68-85f0-c255db698a2f)
Modular RAG flow ([Source](https://github.com/langchain-ai/rag-from-scratch))

Basic RAG, often referred to as Naive RAG, is slightly outdated. Not only is the pipeline too simple, but it also often struggles to understand complex user queries, retrieves irrelevant chunks, and ultimately results in hallucinated responses. Attempting to solve these issues, the state of the art version of RAG called Modular RAG incorporates various modules that can be modified and applied to enhance RAG performance. Langchain defines the modules as follows:
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

[RAG-Fusion](https://arxiv.org/abs/2402.03367)  flow ([Source](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb))

A slightly modified version of Multi-Query is RAG-Fusion, which adds an additional step of ranking the documents before giving the retrieved content as context to the LLM. The ranking process uses ranks the document based on a score called [Reciprocal Rank Fusion (RRF)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf), which is calculated as follows:


$$RRFscore \left(d \in D \right) = \sum_{r \in R} \frac{1}{k + r \left( d \right)}$$


where
- D is the set of all documents
- d is a document
- R is the set of rankers (retrievers)    
- k is a constant (typically 60)
- r(d) is the rank of document d in ranker r 

Here is the code implementation (we again assume that the indexing and loading steps are the same): 

```python
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from langchain_core.runnables import RunnablePassthrough

# RAG-Fusion: Related
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_rag_fusion 
    | ChatOpenAI(temperature=0)
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion

# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    {"context": retrieval_chain_rag_fusion, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"question":question})

```
While all others remain unchanged, the retrieval chain is slightly modified with the addition of the `reciprocal_rank_fusion` function. As the description states, the function takes in multiple documents with certain 'pre-rankings'. The 'pre-rankings' of the retrieved documents are equivalent to the position of the documents in the list, each of which is assigned by the vectorstore. For each unique document, the function calculates a fused score using the RRF algorithm and outputs the list of document-and-score tuples in descending order. Recall that there are multiple queries generated and compared with the document embeddings for retrieval of documents. This increases the likelihood of retrieval of same document(s) with different queries, which would have the fused score cumulate multiple times whenever that specific document is referred to. As for the rankings, the position of the list corresponds to the document's ranking, which follows that the higher the fused score, the more relevant the document is to the original query. By having the `retrieval_chain_rag_fusion` serve as the context, the LLM receives the document content as well as its fused score. 

#### Decomposition

![캡처](https://github.com/user-attachments/assets/f04f410d-f746-4e5d-a64c-e55a8e9f980d)

Least-to-Most Prompting for solving a math question ([Source](https://arxiv.org/pdf/2205.10625))  

![ㅇ](https://github.com/user-attachments/assets/29adf99e-407e-4c62-9323-e4b2c2fffa91)
           
IRCoT ([Source](https://arxiv.org/pdf/2212.10509))

Another technique for query translation is decomposition, where it breaks down the original query into multiple sub-queries. Before we dive into the specifics, we ought to look into some prompting techniques. The idea of breaking down a question into sub questions and solving them sequentially, addressed as Least-to-Most prompting, was first introduced in [this paper](https://arxiv.org/pdf/2205.10625). Prior to the emergence of Least-to-Most, zero-shot prompting, standard prompting, and Chain-of-Thought (CoT) prompting were commonly used. Zero-shot prompting means that there are no demonstrations or examples of questions and answers in the prompt, whereas standard prompting takes a similar question to the original question and pairs it with the answer in the prompt. CoT is an extension of standard prompting, where it provides how the thinking process arrived to the final answer. For instance, suppose we want the LLM to solve the following math question: "Anthony has 12 dollars. He sees that the toy car he wants costs 3 dollars. How many toy cars maximum can he buy?". For CoT prompting, we would provide a similar question like "Sarah has 36 bottles of water. If are 9 marathon athletes, how many bottles of water can each athlete get from Sarah when they are evenly distributed?" and an answer with some reasoning like "Sarah has 36 bottles of water and there are 9 marathon athletes. Since 36 divided by 9 equals 4, each athlete can get 4 bottles of water from Sarah given that they are evenly distributed." In theory, with this kind of prompting, the LLM generates an answer with the following thinking process, allowing for complex reasoning capabilities.   

Going back to Least-to-Most prompting, the paper suggests that based on their experiments, the LLM performed better on symbolic manipulation, compositional generalization, and math reasoning tasks when using Least-to-Most than using CoT. Although we can adopt this approach of sequentially solving the subquestions to RAG, Langchain combines another technique called [Interleaving Retrieval with Chain-of-Thought Reasoning (IRCoT)](https://arxiv.org/pdf/2212.10509). IRCoT interleaves CoT process and retrieves relevant information for each reasoning steps to use the cumulated retrieved information for question answering with CoT reasoning. For better understanding, take a look at the demonstration image above. Notice that the documents get cumulated to carry out the CoT reasoning. We can use this technique to subquestions and create a combination of Least-to-Most and IRCoT. Specifically, by translating the query using Least-to-Most and answering the subquestions recursively with IR-CoT in the retrieval chain, we can create the following flow. 

![image](https://github.com/user-attachments/assets/dcc2bbc5-17b0-4342-b9bb-0cbdac2bad45)

Decomposition flow ([Source](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb))


