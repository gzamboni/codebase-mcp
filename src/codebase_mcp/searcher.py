from pathlib import Path

from openai import OpenAI

from .store import get_client, load_config

TOP_K = 8


def search(query: str, repo_path: str | None = None) -> str:
    config = load_config()

    if repo_path:
        abs_path = str(Path(repo_path).resolve())
        if abs_path not in config:
            return f"Repo not indexed. Run: codebase-mcp index {repo_path}"
        repo_ids = [config[abs_path]["repo_id"]]
    else:
        if not config:
            return "No repos indexed. Run: codebase-mcp index /path/to/repo"
        repo_ids = [v["repo_id"] for v in config.values()]

    openai_client = OpenAI()
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
    )
    query_vector = response.data[0].embedding

    qdrant = get_client()
    all_results = []
    for repo_id in repo_ids:
        try:
            results = qdrant.query_points(
                collection_name=repo_id,
                query=query_vector,
                limit=TOP_K,
            )
            all_results.extend(results.points)
        except Exception:
            continue

    if not all_results:
        return "No results found."

    all_results.sort(key=lambda r: r.score, reverse=True)
    top = all_results[:TOP_K]

    parts = []
    for r in top:
        p = r.payload
        parts.append(
            f"### {p['file']} (lines {p['start_line']}-{p['end_line']}) — score: {r.score:.3f}\n"
            f"```\n{p['text']}\n```"
        )
    return "\n\n".join(parts)
