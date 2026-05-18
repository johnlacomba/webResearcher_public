> python3 -m research_tool --help

usage: research_tool [-h] {research,wiki,repo,query,status,migrate-embeddings,re-ingest,benchmark,eval} ...

Autonomous iterative web research tool with RAG-augmented querying.

positional arguments:
  
    research            Run an autonomous research loop
    wiki                Exhaustively crawl and ingest a wiki/docs site
    repo                Clone and index a Git repo or all repos in a Bitbucket workspace
    query               Query the research database
    status              Show research database statistics
    migrate-embeddings  Re-embed legacy chunks with the current model
    re-ingest           Re-process all stored pages through the current ingest pipeline (generates child chunks, context summaries, and fresh embeddings)
    benchmark           Run or compare benchmark configurations
    eval                Evaluate retrieval quality against a labeled eval set

options:
  -h, --help            show this help message and exit


Quickstart:

Autonomously research a query (using DuckDuckGo), embedding and ingesting the documents into a vector DB:
python3 -m research_tool research --llm ollama --db testing.db --max-depth 1 "what is retrieval augmented generation"

Query a DB:

python3 -m research_tool query --llm ollama --db testing.db "how does RAG work"
