"""CLI entry point for the autonomous web research tool.

Usage:
    python3 -m research_tool research "prompt" [--llm omlx] [--max-depth N] [--db PATH]
    python3 -m research_tool query ["question"] [--llm omlx] [--db PATH]
    python3 -m research_tool wiki <url> [--branch NAME] [--concurrency N] [--db PATH]
    python3 -m research_tool status [--db PATH]
    python3 -m research_tool re-ingest [--llm omlx] [--db PATH]
    python3 -m research_tool benchmark run [--config NAME] [--llm] [--top-k N]
    python3 -m research_tool benchmark compare <file_a> <file_b>
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys

DEFAULT_DB = os.environ.get("RESEARCH_DB", "research.db")


def _make_llm(args: argparse.Namespace):
    """Create an LLMClient from CLI args."""
    from research_tool.brain import LLMClient

    backend = getattr(args, "llm", "claude")
    model = getattr(args, "model", None)
    return LLMClient(backend=backend, model=model)


def _ensure_db_dir(db_path: str) -> None:
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    """Add --llm and --model flags to a subparser."""
    parser.add_argument(
        "--llm", choices=["claude", "omlx", "ollama"], default="claude",
        help="LLM backend: 'claude' (Anthropic API), 'omlx' (local server), or 'ollama'. Default: claude",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name override. For omlx|ollama, auto-detected from server if omitted.",
    )


def cmd_research(args: argparse.Namespace) -> None:
    _ensure_db_dir(args.db)

    from research_tool.brain import ResearchLoop
    from research_tool.store import ResearchStore
    from research_tool.web import Browser

    store = ResearchStore(db_path=args.db)
    llm = _make_llm(args)
    browser = Browser()

    shutdown_requested = False

    def handle_sigint(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            sys.exit(1)
        shutdown_requested = True
        print("\nShutting down gracefully... (press Ctrl+C again to force)", file=sys.stderr)

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        with browser:
            loop = ResearchLoop(
                prompt=args.prompt,
                store=store,
                browser=browser,
                llm_client=llm,
                max_depth=args.max_depth,
                similarity_threshold=args.similarity_threshold,
            )
            result = loop.run()

        print(f"\nResearch complete.", file=sys.stderr)
        print(f"  Depth reached: {result['depth_reached']}", file=sys.stderr)
        print(f"  Total chunks:  {result['total_chunks']}", file=sys.stderr)
        print(f"  Terminated:    {result['terminated_reason']}", file=sys.stderr)
    finally:
        store.close()


def cmd_query(args: argparse.Namespace) -> None:
    from research_tool.brain import QueryEngine

    llm = _make_llm(args)
    engine = QueryEngine(db_path=args.db, llm_client=llm)
    rerank = not getattr(args, "no_rerank", False)

    try:
        if args.question:
            answer = engine.ask(args.question, rerank=rerank)
            if args.json:
                print(json.dumps({"question": args.question, "answer": answer}))
            else:
                print(answer)
        elif not sys.stdin.isatty():
            print("Error: question argument is required in non-interactive mode.", file=sys.stderr)
            sys.exit(2)
        else:
            print("Interactive query mode. Type 'quit' or 'exit' to stop.\n", file=sys.stderr)
            while True:
                try:
                    question = input("question> ")
                except EOFError:
                    break
                if question.strip().lower() in ("quit", "exit", "q"):
                    break
                if not question.strip():
                    continue
                answer = engine.ask(question, rerank=rerank)
                print(f"\n{answer}\n")
    finally:
        engine.close()


def cmd_status(args: argparse.Namespace) -> None:
    from research_tool.brain import QueryEngine

    engine = QueryEngine(db_path=args.db)
    try:
        stats = engine.status()
        sources = engine.list_sources()

        if getattr(args, "json", False):
            output = {**stats, "database": args.db, "sources": sources}
            print(json.dumps(output, indent=2))
        else:
            print(f"Database:       {args.db}")
            print(f"Total pages:    {stats['total_pages']}")
            print(f"Total chunks:   {stats['total_chunks']}")
            print(f"Total images:   {stats['total_images']}")
            print(f"Total links:    {stats['total_links']}")
            print(f"Total searches: {stats['total_searches']}")
            print(f"Last activity:  {stats['last_activity'] or 'none'}")

            if sources:
                print(f"\nSources ({len(sources)}):")
                for s in sources:
                    title = (s['title'] or 'untitled').translate(
                        {c: None for c in range(32) if c not in (10, 13, 9)}
                    )
                    print(f"  {s['url']}  ({title})")
    finally:
        engine.close()


def cmd_migrate_embeddings(args: argparse.Namespace) -> None:
    _ensure_db_dir(args.db)

    from research_tool.store import ResearchStore, migrate_embeddings

    store = ResearchStore(db_path=args.db)
    try:
        if not store.has_unmigrated_chunks():
            print("No unmigrated embeddings found. Database is up to date.", file=sys.stderr)
            return
        print("Migrating embeddings to current model...", file=sys.stderr)
        result = migrate_embeddings(store)
        print(f"\nMigration complete.", file=sys.stderr)
        print(f"  Total:    {result['total']}", file=sys.stderr)
        print(f"  Migrated: {result['migrated']}", file=sys.stderr)
        print(f"  Skipped:  {result['skipped']}", file=sys.stderr)
    finally:
        store.close()


def cmd_reingest(args: argparse.Namespace) -> None:
    _ensure_db_dir(args.db)

    from research_tool.brain import reingest_all_pages
    from research_tool.store import ResearchStore

    store = ResearchStore(db_path=args.db)
    try:
        pages = store.get_all_pages_with_html()
        if not pages:
            print("No pages found in database.", file=sys.stderr)
            return

        print(f"Re-ingesting {len(pages)} pages through current pipeline...", file=sys.stderr)

        llm = None
        if getattr(args, "llm", None):
            llm = _make_llm(args)

        result = reingest_all_pages(
            store,
            llm_client=llm,
            progress_cb=lambda msg: print(msg, file=sys.stderr),
        )

        print(f"\nRe-ingest complete.", file=sys.stderr)
        print(f"  Pages processed:  {result['processed']}/{result['total_pages']}", file=sys.stderr)
        print(f"  Old chunks deleted: {result['chunks_deleted']}", file=sys.stderr)
        print(f"  New chunks created: {result['chunks_created']}", file=sys.stderr)
    finally:
        store.close()



def cmd_benchmark(args: argparse.Namespace) -> None:
    sub = getattr(args, "bench_sub", None)
    if sub == "run":
        _benchmark_run(args)
    elif sub == "compare":
        _benchmark_compare(args)
    else:
        print("Usage: research_tool benchmark {run|compare}", file=sys.stderr)
        sys.exit(1)


def _benchmark_run(args: argparse.Namespace) -> None:
    import time

    from research_tool.benchmarks.configs import generate_matrix, get_config_by_name
    from research_tool.benchmarks.results import build_report, save_report
    from research_tool.benchmarks.runner import BenchmarkRunner, ProgressCallback

    include_llm = getattr(args, "llm_tier", False)

    if args.config:
        names = [n.strip() for n in args.config.split(",")]
        configs = []
        for name in names:
            cfg = get_config_by_name(name, include_llm=include_llm)
            if cfg is None:
                print(f"Error: unknown config '{name}'", file=sys.stderr)
                print("Available configs:", file=sys.stderr)
                for c in generate_matrix(include_llm=include_llm)[:10]:
                    print(f"  {c.name}", file=sys.stderr)
                print(f"  ... ({len(generate_matrix(include_llm=include_llm))} total)", file=sys.stderr)
                sys.exit(1)
            configs.append(cfg)
    else:
        configs = generate_matrix(include_llm=include_llm)

    class CLIProgress(ProgressCallback):
        def __init__(self):
            self._start = time.perf_counter()

        def on_config_start(self, index, total, config_name):
            elapsed = time.perf_counter() - self._start
            if index > 0:
                per_config = elapsed / index
                remaining = per_config * (total - index)
                print(f"[{index + 1}/{total}] {config_name} (elapsed: {elapsed:.0f}s, est. remaining: {remaining:.0f}s)")
            else:
                print(f"[{index + 1}/{total}] {config_name}")

        def on_config_done(self, index, total, result):
            ret_time = result.retrieval_latency.total if result.retrieval_latency else 0.0
            overlap = result.retrieval_metrics.get("mean_text_overlap", 0.0)
            if result.error:
                print(f"  ERROR: {result.error[:80]}")
            else:
                print(
                    f"  done — accuracy {overlap:.1%}, "
                    f"ingestion {result.ingestion_time:.1f}s, "
                    f"retrieval {ret_time:.1f}s, "
                    f"{result.chunk_count} chunks"
                )

    progress = CLIProgress()
    runner = BenchmarkRunner(configs=configs, top_k=args.top_k, progress=progress)

    wall_start = time.perf_counter()
    results = runner.run()
    total_wall = time.perf_counter() - wall_start

    report = build_report(results, total_wall)
    from pathlib import Path
    output_dir = Path(args.output) if args.output else None
    path = save_report(report, output_dir=output_dir)

    print(f"\nResults saved to: {path}")
    print(f"Total time: {total_wall:.1f}s across {len(configs)} configs")

    if not args.json:
        _print_summary_table(report)
        _print_failure_analysis(report)
    else:
        print(json.dumps(report.to_dict(), indent=2))


def _print_summary_table(report) -> None:
    print(f"\n{'Config':<45} {'Overlap':>8} {'Chunks':>7} {'Ingest(s)':>10} {'p50(ms)':>8} {'p95(ms)':>8}")
    print("=" * 90)
    for cfg in report.configs:
        if cfg.error:
            print(f"{cfg.config_name:<45} ERROR: {cfg.error[:35]}")
            continue
        print(
            f"{cfg.config_name:<45} "
            f"{cfg.retrieval.mean_text_overlap:>8.3f} "
            f"{cfg.ingestion.chunk_count:>7} "
            f"{cfg.ingestion.total_time_s:>10.2f} "
            f"{cfg.retrieval.p50_latency_ms:>8.1f} "
            f"{cfg.retrieval.p95_latency_ms:>8.1f}"
        )


def _print_failure_analysis(report) -> None:
    valid_configs = [c for c in report.configs if not c.error and c.queries]
    if not valid_configs:
        return

    num_configs = len(valid_configs)
    num_queries = len(valid_configs[0].queries)

    # Per-query: how many configs failed to retrieve it (overlap == 0)?
    query_fail_counts: dict[int, int] = {}
    for cfg in valid_configs:
        for qi, qd in enumerate(cfg.queries):
            if qd.text_overlap == 0.0:
                query_fail_counts[qi] = query_fail_counts.get(qi, 0) + 1

    # Accuracy by source document
    doc_hits: dict[str, list[float]] = {}
    for cfg in valid_configs:
        for qd in cfg.queries:
            doc_hits.setdefault(qd.source_doc, []).append(qd.text_overlap)

    # Accuracy by difficulty
    diff_hits: dict[str, list[float]] = {}
    for cfg in valid_configs:
        for qd in cfg.queries:
            diff_hits.setdefault(qd.difficulty, []).append(qd.text_overlap)

    # Print accuracy breakdown
    print(f"\n{'Accuracy by document':<35} {'Avg':>8} {'Queries':>8}")
    print("-" * 55)
    ref_queries = valid_configs[0].queries
    for doc in sorted(doc_hits):
        scores = doc_hits[doc]
        avg = sum(scores) / len(scores)
        q_count = sum(1 for q in ref_queries if q.source_doc == doc)
        print(f"  {doc:<33} {avg:>7.1%} {q_count:>8}")

    print(f"\n{'Accuracy by difficulty':<35} {'Avg':>8} {'Queries':>8}")
    print("-" * 55)
    for diff in ["easy", "medium", "hard"]:
        if diff in diff_hits:
            scores = diff_hits[diff]
            avg = sum(scores) / len(scores)
            q_count = sum(1 for q in ref_queries if q.difficulty == diff)
            print(f"  {diff:<33} {avg:>7.1%} {q_count:>8}")

    # List queries that fail across ALL configs (universally missed)
    universal_fails = sorted(
        [(qi, cnt) for qi, cnt in query_fail_counts.items() if cnt == num_configs],
        key=lambda x: x[0],
    )
    # List queries that fail across MOST configs (>75%)
    frequent_fails = sorted(
        [(qi, cnt) for qi, cnt in query_fail_counts.items()
         if cnt >= num_configs * 0.75 and cnt < num_configs],
        key=lambda x: -x[1],
    )

    if universal_fails:
        print(f"\nQueries missed by ALL {num_configs} configs ({len(universal_fails)}):")
        for qi, _ in universal_fails:
            qd = ref_queries[qi]
            print(f"  [{qd.difficulty:<6}] [{qd.source_doc}]")
            print(f"    {qd.query}")

    if frequent_fails:
        print(f"\nQueries missed by >75% of configs ({len(frequent_fails)}):")
        for qi, cnt in frequent_fails[:15]:
            qd = ref_queries[qi]
            print(f"  [{qd.difficulty:<6}] [{qd.source_doc}] ({cnt}/{num_configs} failed)")
            print(f"    {qd.query}")
        if len(frequent_fails) > 15:
            print(f"  ... and {len(frequent_fails) - 15} more")

    total_pass = sum(1 for qi in range(num_queries) if qi not in query_fail_counts)
    total_some_fail = len(query_fail_counts)
    total_all_fail = len(universal_fails)
    print(f"\n  {total_pass}/{num_queries} queries pass in all configs, "
          f"{total_all_fail} never pass, "
          f"{total_some_fail - total_all_fail} fail in some configs")


def _benchmark_compare(args: argparse.Namespace) -> None:
    from pathlib import Path

    from research_tool.benchmarks.results import compare_reports, format_comparison, load_report

    report_a = load_report(Path(args.file_a))
    report_b = load_report(Path(args.file_b))
    diffs = compare_reports(report_a, report_b)

    if args.json:
        from dataclasses import asdict
        print(json.dumps([asdict(d) for d in diffs], indent=2))
    else:
        print(f"Comparing: {args.file_a} (A/baseline) vs {args.file_b} (B/new)")
        print()
        print(format_comparison(diffs))


def cmd_eval(args: argparse.Namespace) -> None:
    from research_tool.eval import (
        evaluate,
        evaluate_answer_quality,
        format_answer_quality_report,
        format_report,
        format_sweep_report,
        load_eval_set,
        rrf_weight_sweep,
    )
    from research_tool.store import HybridIndex, ResearchStore

    store = ResearchStore(db_path=args.db)
    try:
        index = HybridIndex(mode="query")
        index.build_from_store(store)

        eval_set = load_eval_set(args.eval_set)

        if getattr(args, "sweep", False):
            report = rrf_weight_sweep(index, eval_set, top_k=args.top_k, store=store)
            if args.json:
                print(json.dumps(report.to_dict(), indent=2))
            else:
                print(format_sweep_report(report))
            return

        if getattr(args, "answer_quality", False):
            llm = _make_llm(args)
            report = evaluate_answer_quality(
                index, eval_set, llm, top_k=args.top_k, store=store,
            )
            if args.json:
                print(json.dumps(report.to_dict(), indent=2))
            else:
                print(format_answer_quality_report(report))
            return

        report = evaluate(index, eval_set, top_k=args.top_k, store=store)

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(format_report(report))
    finally:
        store.close()


def cmd_repo(args: argparse.Namespace) -> None:
    _ensure_db_dir(args.db)

    from research_tool.repo import (
        RepoIndexer,
        list_workspace_repos,
        parse_repo_url,
        parse_workspace_url,
    )
    from research_tool.store import ResearchStore
    from research_tool.wiki import load_auth_config

    auth_config = load_auth_config(args.auth_config) if args.auth_config else load_auth_config()

    # Determine if this is a workspace URL or a single repo URL
    workspace = parse_workspace_url(args.url)
    repo = parse_repo_url(args.url)

    if workspace:
        platform, ws_slug = workspace
        print(
            f"Listing repos in {platform}/{ws_slug}...",
            file=sys.stderr,
        )
        repos = list_workspace_repos(
            platform, ws_slug, auth_config=auth_config or None,
        )
        if not repos:
            print("No repos found (check credentials and workspace name).", file=sys.stderr)
            return
        print(f"Found {len(repos)} repos.", file=sys.stderr)
    elif repo:
        platform, org, repo_name = repo
        repos = [{"platform": platform, "org": org, "repo_name": repo_name,
                   "full_name": f"{org}/{repo_name}"}]
    else:
        print(f"Unrecognised URL: {args.url}", file=sys.stderr)
        print("Provide a repo URL (github.com/org/repo) or "
              "Bitbucket workspace URL (bitbucket.org/workspace/).",
              file=sys.stderr)
        sys.exit(1)

    store = ResearchStore(args.db)
    indexer = RepoIndexer()
    total_files = 0
    total_chunks = 0
    total_skipped = 0

    for r in repos:
        label = r["full_name"]
        print(f"  Indexing {label}...", file=sys.stderr)
        try:
            stats = indexer.index_repo(
                platform=r["platform"],
                org=r["org"],
                repo_name=r["repo_name"],
                store=store,
                auth_config=auth_config or None,
                branch=args.branch,
            )
            if stats.get("skipped"):
                total_skipped += 1
                print(f"    Unchanged, skipping", file=sys.stderr)
            else:
                total_files += stats["files_indexed"]
                total_chunks += stats["chunks_stored"]
                print(
                    f"    {stats['files_indexed']} files, "
                    f"{stats['chunks_stored']} chunks",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(f"    Failed: {exc}", file=sys.stderr)

    print(
        f"\nRepo indexing complete: {total_files} files, "
        f"{total_chunks} chunks across {len(repos)} repos "
        f"({total_skipped} unchanged, skipped).",
        file=sys.stderr,
    )


def cmd_wiki(args: argparse.Namespace) -> None:
    import asyncio

    _ensure_db_dir(args.db)

    from research_tool.wiki import WikiCrawler, load_auth_config, merge_cli_auth
    from urllib.parse import urlparse

    seed_domain = urlparse(args.url).netloc

    auth_config = {}
    if args.auth_config:
        auth_config = load_auth_config(args.auth_config)
    else:
        auth_config = load_auth_config()

    auth_config = merge_cli_auth(
        auth_config,
        github_token=args.github_token,
        auth_headers=args.auth_header,
        seed_domain=seed_domain,
    )

    shutdown_requested = False

    def handle_sigint(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            sys.exit(1)
        shutdown_requested = True
        print("\nShutting down gracefully... (press Ctrl+C again to force)", file=sys.stderr)

    signal.signal(signal.SIGINT, handle_sigint)

    crawler = WikiCrawler()
    stats = asyncio.run(
        crawler.crawl(
            seed_url=args.url,
            db_path=args.db,
            auth_config=auth_config if auth_config else None,
            concurrency=args.concurrency,
            branch=args.branch,
        )
    )

    print(f"\nWiki crawl complete.", file=sys.stderr)
    print(f"  Pages visited:     {stats.get('visited', 0)}", file=sys.stderr)
    print(f"  New pages:         {stats.get('new_pages', 0)}", file=sys.stderr)
    print(f"  Updated:           {stats.get('updated', 0)}", file=sys.stderr)
    print(f"  Skipped unchanged: {stats.get('skipped_unchanged', 0)}", file=sys.stderr)
    print(f"  Failed:            {stats.get('failed', 0)}", file=sys.stderr)
    print(f"  Repos found:       {stats.get('repos_found', 0)}", file=sys.stderr)
    print(f"  Stale pages:       {stats.get('stale_pages', 0)}", file=sys.stderr)
    print(f"  Elapsed:           {stats.get('elapsed_s', 0):.1f}s", file=sys.stderr)

    # Index discovered repos
    repo_urls = stats.get("repos_found_urls", [])
    if repo_urls:
        from research_tool.repo import RepoIndexer, parse_repo_url
        from research_tool.store import ResearchStore

        print(f"\nIndexing {len(repo_urls)} discovered repos...", file=sys.stderr)
        store = ResearchStore(args.db)
        indexer = RepoIndexer()
        for repo_url in repo_urls:
            parsed = parse_repo_url(repo_url)
            if not parsed:
                print(f"  Skipping unrecognised repo URL: {repo_url}", file=sys.stderr)
                continue
            platform, org, repo_name = parsed
            print(f"  Indexing {platform}/{org}/{repo_name}...", file=sys.stderr)
            try:
                repo_stats = indexer.index_repo(
                    platform=platform,
                    org=org,
                    repo_name=repo_name,
                    store=store,
                    auth_config=auth_config if auth_config else None,
                    branch=args.branch,
                )
                if repo_stats.get("skipped"):
                    print(f"    Unchanged, skipping", file=sys.stderr)
                else:
                    print(
                        f"    {repo_stats['files_indexed']} files, "
                        f"{repo_stats['chunks_stored']} chunks indexed",
                        file=sys.stderr,
                    )
            except Exception as exc:
                print(f"    Failed: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="research_tool",
        description="Autonomous iterative web research tool with RAG-augmented querying.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # research subcommand
    p_research = subparsers.add_parser("research", help="Run an autonomous research loop")
    p_research.add_argument("prompt", help="Research topic or question")
    p_research.add_argument("--max-depth", type=int, default=5, help="Maximum iteration depth (default: 5)")
    p_research.add_argument("--db", default=DEFAULT_DB, help=f"SQLite database path (default: {DEFAULT_DB})")
    p_research.add_argument(
        "--similarity-threshold", type=float, default=0.85,
        help="Cosine similarity threshold for dedup (default: 0.85)",
    )
    _add_llm_args(p_research)
    p_research.set_defaults(func=cmd_research)

    # wiki subcommand
    p_wiki = subparsers.add_parser("wiki", help="Exhaustively crawl and ingest a wiki/docs site")
    p_wiki.add_argument("url", help="Seed URL for the wiki crawl")
    p_wiki.add_argument("--branch", default=None, help="Target branch for repo clones (default: default branch)")
    p_wiki.add_argument(
        "--concurrency", type=int, default=10,
        help="Number of concurrent crawl workers (default: 10)",
    )
    p_wiki.add_argument("--db", default=DEFAULT_DB, help=f"SQLite database path (default: {DEFAULT_DB})")
    p_wiki.add_argument("--auth-config", default=None, help="Path to wiki_auth.yaml config file")
    p_wiki.add_argument("--github-token", default=None, help="GitHub PAT for repo scraping")
    p_wiki.add_argument(
        "--auth-header", action="append", default=None,
        help="Auth header for seed domain (repeatable, format: 'Header: Value')",
    )
    p_wiki.set_defaults(func=cmd_wiki)

    # repo subcommand
    p_repo = subparsers.add_parser(
        "repo",
        help="Clone and index a Git repo or all repos in a Bitbucket workspace",
    )
    p_repo.add_argument(
        "url",
        help="Repo URL (github.com/org/repo) or Bitbucket workspace URL (bitbucket.org/workspace/)",
    )
    p_repo.add_argument("--branch", default=None, help="Branch to clone (default: repo default)")
    p_repo.add_argument("--db", default=DEFAULT_DB, help=f"SQLite database path (default: {DEFAULT_DB})")
    p_repo.add_argument("--auth-config", default=None, help="Path to wiki_auth.yaml config file")
    p_repo.set_defaults(func=cmd_repo)

    # query subcommand
    p_query = subparsers.add_parser("query", help="Query the research database")
    p_query.add_argument("question", nargs="?", default=None, help="Question (omit for interactive mode)")
    p_query.add_argument("--db", default=DEFAULT_DB, help=f"SQLite database path (default: {DEFAULT_DB})")
    p_query.add_argument(
        "--no-rerank", action="store_true", default=False,
        help="Disable cross-encoder reranking (use RRF order only)",
    )
    p_query.add_argument("--json", action="store_true", default=False, help="Output as JSON")
    _add_llm_args(p_query)
    p_query.set_defaults(func=cmd_query)

    # status subcommand (no LLM needed)
    p_status = subparsers.add_parser("status", help="Show research database statistics")
    p_status.add_argument("--db", default=DEFAULT_DB, help=f"SQLite database path (default: {DEFAULT_DB})")
    p_status.add_argument("--json", action="store_true", default=False, help="Output as JSON")
    p_status.set_defaults(func=cmd_status)

    # migrate-embeddings subcommand (no LLM needed)
    p_migrate = subparsers.add_parser(
        "migrate-embeddings", help="Re-embed legacy chunks with the current model"
    )
    p_migrate.add_argument("--db", default=DEFAULT_DB, help=f"SQLite database path (default: {DEFAULT_DB})")
    p_migrate.set_defaults(func=cmd_migrate_embeddings)

    # re-ingest subcommand
    p_reingest = subparsers.add_parser(
        "re-ingest",
        help="Re-process all stored pages through the current ingest pipeline "
             "(generates child chunks, context summaries, and fresh embeddings)",
    )
    p_reingest.add_argument("--db", default=DEFAULT_DB, help=f"SQLite database path (default: {DEFAULT_DB})")
    _add_llm_args(p_reingest)
    p_reingest.set_defaults(func=cmd_reingest)

    # benchmark subcommand
    p_bench = subparsers.add_parser("benchmark", help="Run or compare benchmark configurations")
    bench_sub = p_bench.add_subparsers(dest="bench_sub", required=True)

    p_bench_run = bench_sub.add_parser("run", help="Run benchmark matrix")
    p_bench_run.add_argument("--config", default=None, help="Comma-separated config names (default: full matrix)")
    p_bench_run.add_argument("--top-k", type=int, default=10, help="Number of retrieval results (default: 10)")
    p_bench_run.add_argument("--llm", dest="llm_tier", action="store_true", default=False, help="Include LLM-required tier configs")
    p_bench_run.add_argument("--output", default=None, help="Output directory for results JSON")
    p_bench_run.add_argument("--json", action="store_true", default=False, help="Output full JSON to stdout")
    p_bench.set_defaults(func=cmd_benchmark)

    p_bench_compare = bench_sub.add_parser("compare", help="Compare two benchmark result files")
    p_bench_compare.add_argument("file_a", help="Baseline results JSON")
    p_bench_compare.add_argument("file_b", help="New run results JSON")
    p_bench_compare.add_argument("--json", action="store_true", default=False, help="Output diff as JSON")

    p_eval = subparsers.add_parser("eval", help="Evaluate retrieval quality against a labeled eval set")
    p_eval.add_argument("eval_set", help="Path to eval set JSON file")
    p_eval.add_argument("--db", default=DEFAULT_DB, help=f"SQLite database path (default: {DEFAULT_DB})")
    p_eval.add_argument("--top-k", type=int, default=10, help="Number of results to evaluate (default: 10)")
    p_eval.add_argument("--json", action="store_true", default=False, help="Output as JSON")
    p_eval.add_argument(
        "--answer-quality", action="store_true", default=False,
        help="Evaluate end-to-end answer quality using LLM-as-judge",
    )
    p_eval.add_argument(
        "--sweep", action="store_true", default=False,
        help="Run RRF weight sweep to find optimal weight configuration",
    )
    _add_llm_args(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
