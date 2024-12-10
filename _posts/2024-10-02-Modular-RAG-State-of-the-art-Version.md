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
> The code format for the pipeline depends on the framework on which the pipeline is based (e.g. Langchain, Haystack, LlamaIndex, etc). This post will use sample codes written by [Lance Martin](https://github.com/rlancemartin) from Langchain's RAG from Scratch series to explain the details.

Here is a sample code of Multi-Query: 

```



```
