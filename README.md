Autonomous iterative web research tool with RAG-augmented querying.
    
    usage: research_tool [-h] {research,wiki,repo,query,status,migrate-embeddings,re-ingest,benchmark,eval} ...

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

--------------------------------------------------------------------------------------------------------------

# Build the image
    docker compose build

# Set API keys
    export ANTHROPIC_API_KEY=sk-...
    export OMLX_API_KEY=your-omlx-key

# Research
    docker compose run researcher research "your research question"
    docker compose run researcher research "your research question" --llm omlx
    docker compose run researcher research "your research question" --llm ollama

# Query
    docker compose run researcher query "your question"
    docker compose run researcher query  # interactive mode

# Crawl a wiki
    docker compose run researcher wiki https://docs.example.com

# Index a repo
    docker compose run researcher repo https://github.com/org/repo

# Check status
    docker compose run researcher status

--------------------------------------------------------------------------------------------------------------

# Example research command:
    python3 -m research_tool research --llm ollama --db testing.db --max-depth 1 "what is retrieval augmented generation"

# Example query command:
    python3 -m research_tool query --llm ollama --db testing.db "how does RAG work"
