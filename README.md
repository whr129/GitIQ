# Repo RAG Toolkit

This repository provides a lightweight retrieval-augmented generation (RAG) toolkit tailored for local source-code collections. It lets you index a project, run semantic search over the indexed chunks, and optionally generate answers to natural-language questions by feeding the retrieved context into an LLM.

## Key Features

- **Repository indexing** – walk a codebase, filter by extension, chunk files, and store embeddings in a persistent Chroma vector store.
- **Flexible embeddings** – choose between local sentence-transformer models and the OpenAI embeddings API.
- **Semantic querying** – run cosine-similarity search against the stored chunks to surface relevant snippets quickly.
- **Answer generation** – `repo_rag.generator` combines retrieval results with a templated prompt and calls the OpenAI Responses API to produce concise answers (always ending with “thanks for asking!”).
- **Configurable CLI** – both indexing and querying expose arguments (and optional TOML config support) so you can tailor chunk sizes, file filters, embedding models, and more.

## Getting Started

### Prerequisites

- Python 3.10+
- Access to the command line on macOS, Linux, or Windows
- (Optional) An OpenAI API key if you plan to use the OpenAI embedding backend or the generation workflow

### Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

The requirements install the default sentence-transformer model, the Chroma vector store, and optional OpenAI bindings.

### Environment Variables

Create a `.env` file (already provided in this repo) and export the values before running the CLI:

```bash
source .env
```

At minimum, set `OPENAI_API_KEY` if you intend to use OpenAI embeddings or the generation pipeline. Adding `export` in front of the assignments inside `.env` ensures they are exported when sourced.

## Usage

### 1. Index a Repository

```bash
python -m repo_rag index \
  --repo /path/to/your/project \
  --output /path/to/index_dir \
  --backend sentence-transformer \
  --chunk-size 500 \
  --chunk-overlap 100 \
  --extensions .py .md
```

Important flags:

- `--repo`: root directory to scan (defaults to current working directory).
- `--output`: directory where Chroma persistence files are written.
- `--backend`: `sentence-transformer` (default) or `openai`.
- `--model-name`: override the embedding model identifier.
- `--extensions`: optional whitelist of file suffixes to include.

The command builds (or rebuilds) the persistent vector store at the target output directory.

### 2. Query an Index

```bash
python -m repo_rag query \
  --index /path/to/index_dir \
  --question "How does the Minesweeper reveal logic work?" \
  --backend sentence-transformer \
  --top-k 5
```

This loads the saved store, embeds the question, and prints the top matches with similarity scores. Use the same backend/model combination you used during indexing to avoid representation mismatches.

### 3. Retrieval-Augmented Generation

`repo_rag.generator` bundles indexing (with optional reuse), querying, and answer generation into a single command. Example:

```bash
python -m repo_rag.generator \
  --repo /path/to/your/project \
  --index /path/to/index_dir \
  --question "When you click on a non-mine cell, how is the number revealed?" \
  --output /path/to/logs/minesweeper_run.txt \
  --top-k 5
```

What the command does:

1. Rebuilds the index if `--force-reindex` is set or the target directory is empty; otherwise reuses the persisted store.
2. Runs semantic search to grab the top *k* chunks.
3. Feeds the retrieved chunks into the answer template, calls the OpenAI Responses API (default model `gpt-5`), and prints the answer.
4. Logs the question, ranked snippets, and final answer to the file specified by `--output`.

Optional switches let you override the embedding backend/model, tweak chunk configuration, swap the response model, or supply a custom prompt template. Provide `--openai-api-key` on the command line if you prefer not to export the key globally.

## Configuration via TOML

Both the `index` and `query` subcommands accept a `--config path/to/config.toml` argument. The config file can define shared defaults, for example:

```toml
[embedding]
backend = "sentence-transformer"
model_name = "sentence-transformers/all-MiniLM-L6-v2"

[indexing]
repo = "../my-project"
output = "repo_index/my_project"
chunk_size = 400
chunk_overlap = 80
extensions = [".py", ".md"]

[query]
index = "repo_index/my_project"
top_k = 3
```

Command-line flags always override the config values.

## Project Structure

```
repo_rag/
  __main__.py      # CLI entry point (index/query)
  pipeline.py      # High-level orchestration for indexing/querying
  chunker.py       # Document chunking utilities
  loader.py        # Repository file loader with filtering controls
  embeddings.py    # Embedding backend abstractions (OpenAI & sentence-transformers)
  vectorstore.py   # Chroma-backed persistent vector store
  generator.py     # End-to-end RAG utility combining retrieval and generation
  config.py        # TOML config parser
output/            # Sample log files from generator runs
repo_index/        # Example persistent Chroma stores
requirements.txt   # Python dependencies
```

## Tips

- Use a virtual environment to keep dependencies isolated.
- For large repositories, adjust `--chunk-size` and `--chunk-overlap` to balance context granularity and indexing time.
- When switching embedding backends or models, rebuild the index to avoid mixing incompatible vector representations.
- Add new extensions to the whitelist or edit `loader.py` if you need broader file coverage.

## License

This project is distributed under the terms of the MIT License. See [LICENSE](LICENSE) for details.

