Usage:

Autonomously research a query (using DuckDuckGo), embedding and ingesting the documents into a vector DB:
python3 -m research_tool research --llm ollama --db testing.db --max-depth 1 "what is retrieval augmented generation"

Query a DB:
python3 -m research_tool query --llm ollama --db testing.db "how does RAG work"
