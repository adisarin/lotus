# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LOTUS (LLMs Over Text, Unstructured and Structured Data) is a Python framework for LLM-powered data processing with a Pandas-like API. It introduces **semantic operators** that extend relational operators to work with unstructured data using natural language expressions (langex).

## Common Commands

```bash
# Install dependencies (uses uv)
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_filter.py

# Run linting and formatting
uv run pre-commit run --all-files

# Run type checking
uv run mypy lotus/
```

### Environment Variables for Testing
```bash
export ENABLE_OPENAI_TESTS="true"
export ENABLE_LOCAL_TESTS="true"
export OPENAI_API_KEY="<your key>"
```

## Architecture

### Core Components

**Semantic Operators** (`lotus/sem_ops/`): The primary API for data processing
- `sem_map` - Transform records with natural language projections
- `sem_filter` - Filter by natural language predicates
- `sem_extract` - Extract structured attributes from text
- `sem_agg` - Aggregate/summarize across records
- `sem_topk` - Sort by semantic criteria
- `sem_join` - Join datasets on natural language predicates
- `sem_search` - Vector-based semantic search
- `sem_sim_join` - Similarity-based joins
- `sem_cluster_by`, `sem_dedup`, `sem_partition_by`

**Language Model Interface** (`lotus/models/lm.py`):
- Unified LLM interface via LiteLLM (supports OpenAI, Ollama, vLLM, etc.)
- Batch processing, rate limiting, token counting, cost tracking, caching

**AST & Lazy Evaluation** (`lotus/ast.py`):
- `LazyFrame` - Records semantic operator pipelines without executing
- Supports predicate pushdown optimization (pandas filters before sem_filters)
- `print_tree()` / `print_optimized_tree()` for visualization
- `execute(optimize=True)` materializes the DataFrame

**Vector Stores** (`lotus/vector_store/`): Pluggable backends (FAISS, Qdrant, Weaviate)

**Settings** (`lotus/settings.py`): Global configuration via `lotus.settings.configure(lm=lm)`

### Key Patterns

**Langex (Language Expressions)**: Natural language strings parameterized by column names in brackets:
```python
df.sem_filter("{Course Name} requires math")
df.sem_join(skills_df, "Taking {Course Name} will help learn {Skill}")
```

**Model Configuration**:
```python
from lotus.models import LM
lm = LM(model="gpt-4o")  # or "ollama/llama3.2", "hosted_vllm/..."
lotus.settings.configure(lm=lm)
```

**LazyFrame Usage**:
```python
from lotus.ast import LazyFrame
lf = LazyFrame(df, name="data")
lf = lf.sem_filter("...").sem_map("...")
lf.print_tree()           # logical plan
lf.print_optimized_tree() # physical plan
result = lf.execute()     # materialize
```

## Testing Guidelines

- Two test suites: `.github/tests/` (CI essentials) and `tests/` (comprehensive)
- Avoid assertions on exact LLM outputs; assert on schema/structure instead
- Mock external dependencies
