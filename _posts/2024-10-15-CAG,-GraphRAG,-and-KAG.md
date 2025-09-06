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

[!Note]
In practice you don't need RDF to benefit from "KG thinking". Property-graph stacks like Neo4j/Cypher provide similar utility for RAG/KG hybrids.

