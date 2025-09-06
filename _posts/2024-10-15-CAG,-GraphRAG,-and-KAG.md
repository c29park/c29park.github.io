## CAG, GraphRAG, and KAG

Ordinary RAG (vector search -> stuff passages into context) is great for many tasks, but it struggles with multi-hop reasoning, strict logic like time or numbers, and precise entity relationships. 
Below I summarize the first proposal or seminal sources for four adjacent ideas: 
+ Knowledge Graphs (KGs)
+ Knowledge-Augmented Generation (KAG)
+ GraphRAG
+ Cache-Augmented Generation (CAG)

I will also explain when each makes sense compared to the basic RAG.

------------------------------------------------------------
### What's a Knowledge Graph (KG)?

<img width="899" height="714" alt="image" src="https://github.com/user-attachments/assets/b5a3c782-37ff-4844-9c23-28bfc731232e" />

A knowledge graph encodes entities and relations (nodes/edges) under a certain schema, enabling queries and reasoning that go beyond some fuzzy similarities. The term "Knowledge Graph" rose to prominence with Google's 2012 announcement, which framed it as ["things, not strings"](https://blog.google/products/search/introducing-knowledge-graph-things-not/).

Academically, the strongest early exemplars are DBpedia (extracting structured facts from Wikipedia; ISWC/ASWC 2007) and YAGO (unifying Wikipedia with WordNet; WWW 2007). These are widely cited as foundational KG efforts. 

> [!Note]
> In practice you don't need RDF to benefit from "KG thinking". Property-graph stacks like Neo4j/Cypher provide similar utility for RAG/KG hybrids.

### [Reasoning on Graphs (RoG)](https://arxiv.org/pdf/2310.01061)

<img width="1237" height="715" alt="image" src="https://github.com/user-attachments/assets/2f51c75e-dd35-45ac-8664-ff25d40228dc" />

Here's an example application of how Knowledge graphs could be applied for reasoning with Large Language Models (LLMs). 
Basically how RoG works is that it makes the model plan on the graph before answering. Instead of only retrieving facts, it prompts the LLM to propose relation-path plans like (Joe Biden --> Scranton) that exist in the KG, then pulls solid reasoning paths that match those relations, giving it to the LLM have it reason over them. The outcome, according to the paper, is answers that are both faithful(grounded by the KG) and interpretable (the path makes the reasoning process explainable). 

Check out their github [here](https://github.com/RManLuo/reasoning-on-graphs) to download and test the code. You would need an insane amount of GPU computing power to test them all (hehe). 

### [Knowledge-Augmented Generation (KAG)](https://arxiv.org/pdf/2409.13731)
<img width="1012" height="448" alt="image" src="https://github.com/user-attachments/assets/32e6a60e-d885-4f88-a9b5-c10f6ca122c0" />

KAG is another implementation of KG + LLM, which is more similar to RAG than RoG. KAG explicitly integrates a KG with LLM generation to improve logic-aware, multi-hop reasoning. The initial comprehensive "KAG" proposal (link on the heading) lays out five pillars: LLM-friendly KG representations, mutual indexing between KG and source chunks, logic-guided hybrid reasoning, semantic alignment, and model capability enhancement. They report significant gains over RAG on multihop QA (2Wiki, HotpotQA) and production deployments in e-gov/e-health.

If you need explicit temporal, numeric, or rule-based constraints and path-based reasoning across facts, this is the go-to. RAG alone just tends to miss or blur these things, as it only bases its reasoning on retrieved "relevant" documents. 

### [GraphRAG](https://arxiv.org/pdf/2404.16130)
<img width="896" height="584" alt="image" src="https://github.com/user-attachments/assets/e792a50d-7f6c-485d-8d7c-5a4c61121f42" />

GraphRAG proposes building and using a graph over some corpus like entity-relation graph or some community structure, for example and retrieving subgraphs or graph-aware summaries instead of just flat chunks. The first public proposal paper (linked to the heading) describes graph construction from private corpora and a query-focused summarization workflow that scales beyond typical QFS limits.

If you want better retrieval structure and higher-quality context windows without fully adopting a rule-heavy KAG stack, this can be a simple alternative to KAG. 

### [Cache-Augmented Generation (CAG)](https://arxiv.org/html/2412.15605v1)

This one is pretty funny. The original paper itself advertises the new technique by highlighting "Don't Use RAG" in its paper title. CAG argues that with long-context LLMs, many knowledge tasks can preload a stable corpus and cache model state, avoiding per-query retrieval latency/complexity. The paper shows scenarios where CAG matches or exceeds RAG while simplifying systems, especially when the knowledge base is small/stable.

Basically just use this when you are too lazy to initialize a DB for RAG every time you try to do the reasoning with an LLM. If you need to give repeated queries and latency improvement and operational simplicity, this one's the way to go. 







