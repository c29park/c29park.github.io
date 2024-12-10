## Basics of Retrieval Augmented Generation (RAG)

RAG is a very common approach that can enhance an LLM's response by providing the LLM with external factual knowledge in the generation process.

---

### Introduction

Trained on large amounts of data, Large Language Models (LLMs) have demonstrated its outstanding ability in such common NLP tasks as text generation, analysis, and question answering. However, the pre-trained language models still have their limitations when it comes to accurately processing and using the factual knowledge they store. To exemplify this, consider a case where a user asks an LLM to describe the historical event when Donald Trump declared war on Germany immediately after Adolf Hitler shot him with his Nerf gun. Thinking that the event genuinely happened, the LLM would answer with its own detailed crafted plot. This is an example of the notorious hallucination issue. Hallucination is where an LLM generates responses that are non-sensical or misleading. In addition to the hallucination problem, since LLMs do not have their trained data updated time after time, they have a certain knowledge cutoff and tend to produce outdated information as a result. First proposed in [this paper](https://arxiv.org/pdf/2005.11401), Retrieval Augmented Generation (RAG) resolves both prominent issues to a certain degree by allowing LLM's access to newly retrieved information.         

### Basic RAG

![image](https://github.com/user-attachments/assets/c026a29e-a29f-44b4-8c20-9c3354232eb9) 
Basic RAG flow ([Source](https://blog.langchain.dev/agentic-rag-with-langgraph/))

The general idea of RAG is to retrieve information from external data sources like pdf documents and feed it to an LLM to use it for more accurate response generation. To breakdown the pipeline into a few steps, RAG consists of the following steps: 
1. Loading - Loading external sources (documents)
2. Indexing - Formatting documents and converting them to embeddings
3. Retrieval - Retrieving relevant document embedding(s)
4. Generation - LLM's generation with the relevant context

Now that we have a general idea of how RAG works with each step, let's look into the details in terms of the code.   
> [!Note]
> The code format for the pipeline depends on the framework on which the pipeline is based (e.g. Langchain, Haystack, LlamaIndex, etc). This post will use sample codes written by [Lance Martin](https://github.com/rlancemartin) from [Langchain's RAG from Scratch series](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) to explain the details. 

Here is a sample code of Basic RAG:

```python
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

#### Loading ####
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

#### INDEXING ####
# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
rag_chain.invoke("What is Task Decomposition?")
```

#### Loading

Langchain provides various `document_loaders` to use according to the document type. In this case, `WebBaseLoader` is used to load a blog post from the web along with `BeautifulSoup` to parse the post's content, title, and header. If the document to load were located in a local drive, a document loader like `UnstructuredFileLoader` would be used.

#### Indexing

Earlier, indexing step was introduced as the combination of formatting and embedding. The formatting part is equivalent to the splitting part of the code above. As it is written, the loaded document is split up into chunks by a `text_splitter`. Similar to `document_loaders`, Langchain also has various `text_splitters`, one of which is `RecursiveCharacterTextSplitter` used in the code. The `RecursiveCharacterTextSplitter` recursively splits the document until the specified `chunk_size` is reached. `Chunk_overlap` can also be set to control the number of characters that lie between two adjacent chunks of text. The characters that the splitter uses are `['\n\n', '\n', ' ', '' ]`. Aside from the commonly used `RecursiveCharacterTextSplitter`, there are many other techniques that can be utilized for splitting, such as semantic splitting, agentic splitting, and etc. For more information about these other methods, take a look at [this video](https://www.youtube.com/watch?v=8OJC21T2SL4). 

Embedding is a numerical representation (vector) of a document chunk, which in this case, is made and stored by a database called `ChromaDB`. As written above, the database could take in embedding types like `OpenAIEmbeddings` as its parameters. With the `as_retriever()` function, the database can act as a retriever that would retrieve relevant document chunks from the embedding space that the database created. 

#### Retrieval

Now that we have created an embedding space and stored embeddings in a vectorstore, the question remains for how to perform the retrieval. The goal is to retrieve document chunks relevant to the user query. Therefore, the retrieval process involves similarity search, which finds embeddings of document chunks similar to the embedding of a question. Some examples of vector similarity search are cosine similarity, max marginal relevance, hierarchical navigable small world (HNSW), and etc. The default search methods depend on the vectorstore, and the search type can indeed be specified. Top-k retrieval is widely used along with the search algorithms, which is essentially retrieving the top k number of embeddings. 

#### Generation

For the final step of generation, certain prompt engineering is adopted to allow the LLM to understand its duty for RAG. For the RAG chain, it is noticeable that the dictionary containing context and question key values, `prompt`, `llm`, and `StrOutputParser` make a single chain, where `|` is an operator that connects two adjacent elements as a chain. The `context` key value is the retrieved document chunks formatted by `format_docs` function whereas the question is passed by the `RunnablePassthrough()`. Then using the dictionary, the `context` key value and the `question` key value act as inputs of the pre-existing RAG prompt. The whole RAG prompt is finally given to the LLM, yielding an answer to the question. 

This is a typical set up of a basic RAG pipeline, which nowadays is rarely utilized and almost considered obsolete. 
