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
> The code format for the pipeline depends on the framework on which the pipeline is based (e.g. Langchain, Haystack, LlamaIndex, etc). This post will use sample codes written by [Lance Martin](https://github.com/rlancemartin) from [Langchain's RAG from Scratch series](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) to explain the details. ([Code Source](https://github.com/langchain-ai/rag-from-scratch))

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

Going back to Least-to-Most prompting, the paper suggests that based on their experiments, the LLM performed better on symbolic manipulation, compositional generalization, and math reasoning tasks when using Least-to-Most than using CoT. Although we can adopt this approach of sequentially solving the subquestions to RAG, Langchain combines another technique called [Interleaving Retrieval with Chain-of-Thought Reasoning (IRCoT)](https://arxiv.org/pdf/2212.10509). IRCoT interleaves CoT process and retrieves relevant information for each reasoning steps to use the cumulated retrieved information for question answering with CoT reasoning. For better understanding, take a look at the demonstration image above. Notice that the documents get cumulated to carry out the CoT reasoning. We can use this technique to subquestions and create a combination of Least-to-Most and IRCoT. Specifically, by translating the query using Least-to-Most and answering the subquestions recursively with IRCoT in the retrieval chain, we can create the following flow. 

![image](https://github.com/user-attachments/assets/dcc2bbc5-17b0-4342-b9bb-0cbdac2bad45)

Decomposition flow ([Source](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb))

Based on the flow, the code implementation below can be created.

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser

#Decomposition
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)

# LLM
llm = ChatOpenAI(temperature=0)

# Chain
generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

# Run
question = "What are the main components of an LLM-powered autonomous agent system?"
questions = generate_queries_decomposition.invoke({"question":question})

# Prompt
template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

decomposition_prompt = ChatPromptTemplate.from_template(template)

def format_qa_pair(question, answer):
    """Format Q and A pair"""
    
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()

# llm
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

q_a_pairs = ""
for q in questions:
    
    rag_chain = (
    {"context": itemgetter("question") | retriever, 
     "question": itemgetter("question"),
     "q_a_pairs": itemgetter("q_a_pairs")} 
    | decomposition_prompt
    | llm
    | StrOutputParser())

    answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
    q_a_pair = format_qa_pair(q,answer)
    q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
```

The first prompt for decomposition instructs the LLM to break down the single input question into multiple subquestions. 
The `generate_queries_decomposition` then uses the LLM to generate subquestions with separation by new lines. An example of set of subquestions generated from the specific query "What are the main components of an LLM-powered autonomous agent system?" is `['1. What is an LLM-powered autonomous agent system?', '2. What are the key components of an autonomous agent system?', '3. How does LLM technology enhance the capabilities of an autonomous agent system?' ]`. As mentioned in the prompt, the subquestions are answerable in isolation. However, the goal of IRCoT seems to be slightly unsatisfied with the code above. Although the algorithm above recursively answers the questions step by step by adding `q_a_pairs` as the main context to answer the final question, it defeats the purpose of sequentially answering the questions one by one since the questions are simply answerable in isolation and yet do not build upon each other. Notice that the questions are numbered. This is the order of the questions being answered, meaning that the answer of the third question based on the context with `q_a_pairs` of questions 1 and 2 would be the final output of the original query. Considering this, this prompt does not incorporate the importance of IRCoT's original goal of recursive and sequential answering. Therefore, we could direct the LLM to produce the list of questions such that the order of the questions generated ensures that the questions' answers can build up and help arrive to the final conclusion. We might also use an LLM-as-judge to strengthen the accuracy and relevance of the questions generated, where we use another LLM to judge the questions generated - whether the questions generated form the "build up process" and whether the order satisfies the "build up process". Indeed, there is a computational latency issue if this is added. Just a thought here.

#### Step Back prompting

![wth](https://github.com/user-attachments/assets/fd3c6a88-9dc7-49a8-830c-9fe9e9189f59)
Illustration of Step back prompting ([Source](https://arxiv.org/abs/2310.06117))

Previously, we looked at the decomposition technique, where we translated a single input question to multiple subqueries for better context writing. As the image above and its name suggest, as opposed to decomposition, step back prompting goes more abstract with the input question. For instance, suppose that we received a highly specific input question "What were the main drivers of customer dissatisfaction for Air Canada flights departing from Toronto to Vancouver between July and August 2023?". Then, the step back question of this would be "What are common factors contributing to customer dissatisfaction for flights operated by Air Canada?". We notice that we can formulate a more generalized question to simplify the original question. How would we write the prompt so that the LLM could properly generate a more simplified question? To overcome this challenge, we typically add a few examples of original question and step back question pair so that the LLM learns how the step back question should be generated. This prompting technique of adding a few examples for better generation is called the few-shot prompting. For greater understanding, let's look at Lance's code for step back prompting:

```python
# Few Shot Examples
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ]
)

generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
question = "What is task decomposition for LLM agents?"
generate_queries_step_back.invoke({"question": question})

# Response prompt 
response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

chain = (
    {
        # Retrieve context using the normal question
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        # Retrieve context using the step-back question
        "step_back_context": generate_queries_step_back | retriever,
        # Pass on the question
        "question": lambda x: x["question"],
    }
    | response_prompt
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)

chain.invoke({"question": question})
```

First take a look at the prompt formatting. We first construct a list of example dictionaries that consist of input and output questions. Subsequently, we assign roles for both input and output questions, specifying that the input question comes from a human and that the output question is generated by AI. Finally, we form it into a completed few shot prompt and combine with the main instruction for generating a step back question. A sample output by the `generate_queries_step_back` chain for the input question "What is task decomposition for LLM agents?" is "What is task decomposition?". For the response prompt, the `normal_context`  retrieved with the original question, the `step_back_context` retrieved with the step back question, and the original question is provided. By doing so, the LLM could generate a response with a normal context and a broader context, which allows for more organized response generation. 

#### HyDE

![image](https://github.com/user-attachments/assets/28218584-6806-470b-aa07-993dc4ec84cc)
([Source](https://arxiv.org/abs/2212.10496))

Above query translation methods rely on either abstraction or paraphrasing. This method named HyDE, however, makes use of neither of them. Surprisingly, HyDE (Hypothetical Document Embedding) enables retrieval without the question embedding. As it could be assumed from the name and the figure above, HyDE is a process of transforming the query into a relevant hypothetical document and using it to retrieve real related documents. The method was first introduced in [this paper](https://arxiv.org/pdf/2212.10496), where there was a major problem with dense retrieval -- the standard retrieval method that we are most familiar with. Briefly, the problem was with the similarity measurement and the encoder for the question. Specifically, dense retrieval models define the similarity measurement as the following: 

$$sim\left(q,d\right) = \langle enc_q\left(q\right),enc_d\left(d\right) \rangle = \langle v_q, v_d \rangle$$

where 
+ q = question
+ d = document
+ $$enc_q$$ = encoder function for question 
+ $$enc_d$$ = encoder function for document
+ $$v_q$$ = question embedding
+ $$v_d$$ = document embedding

This equation simply suggests that the conventional dense retrieval's similarity measurement is the inner product between the question embedding and the document embedding. Since inner product can only be  computed when the two vectors are of the same dimension, each encoder has to map the corresponding entity to a vector of the same dimension. It can be mathematically written like this: $$v_q, v_d \in \mathbb{R}^k, k \in \mathbb{Z}^+$$. 
If we were to make the retrieval zero-shot, we need to somehow define the two mapping functions without looking at the queries, the documents, or any relevance judgement. The encoder functions also have to map the query and the document to the same embedding space and have the inner product represent the relevance of the document to the query. On top of that, when it comes to retrieving a relevant document and a question, their semantic difference might hinder an accurate retrieval process. 

To resolve this, we take the question and transform it to a document so that document-document comparison can be made. Here is the code implementation: 
```python
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# HyDE document generation
template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
prompt_hyde = ChatPromptTemplate.from_template(template)

generate_docs_for_retrieval = (
    prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser() 
)

# Run
question = "What is task decomposition for LLM agents?"
generate_docs_for_retrieval.invoke({"question":question})

# Retrieve
retrieval_chain = generate_docs_for_retrieval | retriever 
retrieved_docs = retrieval_chain.invoke({"question":question})

# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"context":retrieved_docs,"question":question})
```

As you can see in the code, the =`generate_docs_for_retrieval` chain takes the input question and converts it into a passage that can possibly solve the question. Then, we make the `retrieval_chain` such that when the question is fed to the chain, relevant documents to the hypothetical document can be retrieved. As always, we use the retrieved documents as the overall context with which the LLM generates response. 

### Routing

![image](https://github.com/user-attachments/assets/238c360c-bd10-405f-9c1b-6836fb026f1c)

([Source](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_10_and_11.ipynb))

Previously, we looked at the techniques for transforming a query to enhance the retrieval of relevant documents. Routing is a stage that comes after the Query Translation component, where it has the LLM route the transformed query to either a desirable database or a prompt for the LLM to use. The routing process for choosing a desirable database to use based on the question is called **Logical routing**. **Semantic routing**, on the other hand, is choosing a prompt for LLM to use for answer generation based on the similarity between the prompt embeddings and the question embedding.

#### Logical Routing
![image](https://github.com/user-attachments/assets/2d918659-f7c8-48b5-9c57-b08ae9f7f095)

Logical Routing flow ([Source](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_10_and_11.ipynb))
![image](https://github.com/user-attachments/assets/a0bc95ed-2a20-417a-a60b-c8b6058e092a)
Detailed code flow ([Source](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_10_and_11.ipynb))

Logical routing is a process where we instruct an LLM to classify the relevant database for retrieval based on the input question. For instance, suppose you have a graph database that stores knowledge graph of casts of *Squid Game* and a vector database that contains overall summary of the series. When the input question is something like "who played Ki-hoon?", the LLM would route the query to the graph database to answer with more relevant information. In the "RAG from Scratch" series, Lance simulated this routing technique with choosing relevant programming languages based on a query written in a certain programming language:
```python
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )

# LLM with function call 
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery)

# Prompt 
system = """You are an expert at routing a user question to the appropriate data source.

Based on the programming language the question is referring to, route it to the relevant data source."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Define router 
router = prompt | structured_llm

question = """Why doesn't the following code work:

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""

result = router.invoke({"question": question})

def choose_route(result):
    if "python_docs" in result.datasource.lower():
        ### Logic here 
        return "chain for python_docs"
    elif "js_docs" in result.datasource.lower():
        ### Logic here 
        return "chain for js_docs"
    else:
        ### Logic here 
        return "golang_docs"

from langchain_core.runnables import RunnableLambda

full_chain = router | RunnableLambda(choose_route)
```


In the code above, `python_docs`, `js_docs`, and `golang_docs` are provided as the available options for choosing the right programming language. The `structured_llm` is instructed to choose from the literals via `RouteQuery` object and to output the JSON string corresponding to the literal. With the prompt and the `structured_llm` chained together as `router`, it prints out the relevant data source. If we were to print out `result`, it would give `python_docs`. Additionally, `choose_route` is essentially a pretty print function that prints the output in a formatted matter. The above flow describes what the code specifically achieves.

#### Semantic Routing
![image](https://github.com/user-attachments/assets/b9732cb4-ea59-4879-bbb0-74cc9d18f4b3)

Semantic Routing flow ([Source](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_10_and_11.ipynb))

Instead of mapping the query to a certain database, we could also map it to a relevant prompt that the LLM needs to use for better response. To map the query to a certain prompt, we embed both the question and the prompts and do a similarity search. This is called semantic routing. Take a look at the code implementation: 

```python
from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Two prompts
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}"""

# Embed prompts
embeddings = OpenAIEmbeddings()
prompt_templates = [physics_template, math_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

# Route question to prompt 
def prompt_router(input):
    # Embed question
    query_embedding = embeddings.embed_query(input["query"])
    # Compute similarity
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    # Chosen prompt 
    print("Using MATH" if most_similar == math_template else "Using PHYSICS")
    return PromptTemplate.from_template(most_similar)


chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | ChatOpenAI()
    | StrOutputParser()
)

print(chain.invoke("What's a black hole"))
``` 

As written in the code, there are two prompts: one that instructs the LLM to use physics for the response and the other that instructs the LLM to use math. Then, we embed both prompts as well as the question and compute the cosine similarity to get the most similar prompt. A sample output of the line `print(chain.invoke("What's a black hole"))` is "Using PHYSICS
A black hole is a region in space where the gravitational pull is so strong that nothing, not even light, can escape from it. It is formed when a massive star collapses in on itself. The boundary surrounding a black hole is called the event horizon. Beyond the event horizon, the gravitational pull is so intense that even time and space are distorted. Black holes are some of the most mysterious and fascinating objects in the universe."

### Query Construction (Structuring)

![image](https://github.com/user-attachments/assets/7e52f995-4948-4567-a492-d9a6231cfd92)

Once we know the database to which the question is to be routed, we could take an additional step of structuring the query -- converting the natural language query into a domain (database) specific language for better retrieval. In the case of routing a question to a vector database, we could transform the query to metadata filters to search for specific document chunks using the vector database's metadata fields. 

![image](https://github.com/user-attachments/assets/3d844659-37ec-4fc4-8fcd-90b10c7ee567)

Let's take Youtube videos as an example. We could simulate query structuring by using metadata filters of youtube videos and the LLM to output the relevant metadata field values. Notice that the flow above is almost identical to the logical routing flow. Indeed, they share the same method of code implementation: 

```python
from langchain_community.document_loaders import YoutubeLoader

docs = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=pbAd8O1Lvm4", add_video_info=True
).load()

docs[0].metadata
```    

If we run this code, we would get the following output: 

```python
{'source': 'pbAd8O1Lvm4',
 'title': 'Self-reflective RAG with LangGraph: Self-RAG and CRAG',
 'description': 'Unknown',
 'view_count': 11922,
 'thumbnail_url': 'https://i.ytimg.com/vi/pbAd8O1Lvm4/hq720.jpg',
 'publish_date': '2024-02-07 00:00:00',
 'length': 1058,
 'author': 'LangChain'}
```
Now we know that a Youtube video has the following fields. 

```python
import datetime
from typing import Literal, Optional, Tuple
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class TutorialSearch(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    content_search: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )
    title_search: str = Field(
        ...,
        description=(
            "Alternate version of the content search query to apply to video titles. "
            "Should be succinct and only include key words that could be in a video "
            "title."
        ),
    )
    min_view_count: Optional[int] = Field(
        None,
        description="Minimum view count filter, inclusive. Only use if explicitly specified.",
    )
    max_view_count: Optional[int] = Field(
        None,
        description="Maximum view count filter, exclusive. Only use if explicitly specified.",
    )
    earliest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Earliest publish date filter, inclusive. Only use if explicitly specified.",
    )
    latest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Latest publish date filter, exclusive. Only use if explicitly specified.",
    )
    min_length_sec: Optional[int] = Field(
        None,
        description="Minimum video length in seconds, inclusive. Only use if explicitly specified.",
    )
    max_length_sec: Optional[int] = Field(
        None,
        description="Maximum video length in seconds, exclusive. Only use if explicitly specified.",
    )

    def pretty_print(self) -> None:
        for field in self.__fields__:
            if getattr(self, field) is not None and getattr(self, field) != getattr(
                self.__fields__[field], "default", None
            ):
                print(f"{field}: {getattr(self, field)}")

system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a database query optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(TutorialSearch)
query_analyzer = prompt | structured_llm
```

The code above defines a schema for generating structured queries, where it allows to perform `title_search` and `content_search` as well as range filtering for `view count`, `publication date`, and `length`. As always, the system prompt is given that specifies the action of returning a database query to retrieve the most relevant document chunks. If we were to feed the LLM with an input query such as "videos that are focused on the topic of chat langchain that are published before 2024", then the output would be ```content_search: chat langchain
title_search: chat langchain
earliest_publish_date: 2024-01-01``` with the ```pretty_print``` applied to the output. Connecting this metadata filtering technique to databases takes a process called self-querying. To see the actual code implementation of self-querying, visit [here](https://python.langchain.com/docs/how_to/self_query/). 

