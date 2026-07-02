---
name: db-inspect
description: Inspect the project's SQLite databases (ingest_jobs, agent_tasks, matters, memory) and ChromaDB collections for debugging.
user-invocable: true
argument-hint: [database name or "all"]
---

# Database Inspection

Quick inspection of the legal document assistant's persistence layer.

## Databases

| File | Purpose |
|------|---------|
| `data/ingest_jobs.sqlite3` | Document ingestion job tracking |
| `data/agent_tasks.sqlite3` | Agent task & event history |
| `data/matters.sqlite3` | Legal matters, findings, artifacts |
| `data/memory.sqlite3` | User memories, conversations, feedback |

## Operations

### List all tables across all DBs

```bash
for db in data/ingest_jobs.sqlite3 data/agent_tasks.sqlite3 data/matters.sqlite3 data/memory.sqlite3; do
  echo "=== $db ==="
  python -c "
import sqlite3
conn = sqlite3.connect('$db')
tables = [r[0] for r in conn.execute(\"SELECT name FROM sqlite_master WHERE type='table' ORDER BY name\").fetchall()]
for t in tables:
    count = conn.execute(f'SELECT COUNT(*) FROM \"{t}\"').fetchone()[0]
    print(f'  {t}: {count} rows')
"
  echo
done
```

### Inspect a specific DB's table

```bash
python -c "
import sqlite3, json
conn = sqlite3.connect('data/<db_name>.sqlite3')
cursor = conn.cursor()
cols = [c[1] for c in cursor.execute('PRAGMA table_info(<table_name>)').fetchall()]
cursor.execute('SELECT * FROM <table_name> ORDER BY rowid DESC LIMIT 10')
for row in cursor.fetchall():
    print(dict(zip(cols, row)))
"
```

### Check ChromaDB collections

```bash
python -c "
import chromadb
client = chromadb.PersistentClient(path='data/vector_store', settings=chromadb.Settings(anonymized_telemetry=False))
for col in client.list_collections():
    print(f'{col.name}: {col.count()} documents')
"
```
