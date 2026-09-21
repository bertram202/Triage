import hashlib
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import sqlite_vec
from langchain_ollama import OllamaEmbeddings

from app.config import settings


@dataclass
class Policy:
    doc_id: str
    title: str
    body: str


def load_policies() -> list[Policy]:
    policies = []
    for path in sorted(Path(settings().policies_dir).glob("*.md")):
        title, _, body = path.read_text(encoding="utf-8").strip().partition("\n")
        policies.append(Policy(path.stem, title.lstrip("# ").strip(), body.strip()))
    return policies


def chunks(policy: Policy) -> list[str]:
    parts = []
    for para in re.split(r"\n\s*\n", policy.body):
        para = para.strip()
        if not para:
            continue
        parts.append(para)
        lines = para.splitlines()
        items = [ln[2:].strip(" ;.") for ln in lines if ln.startswith("- ")]
        if items:
            lead = " ".join(ln for ln in lines if not ln.startswith("- ")).rstrip(":")
            parts += [f"{lead}: {item}." for item in items]
    return [f"{policy.title}. {p}" for p in parts]


@lru_cache
def embedder() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=settings().embed_model, base_url=settings().embed_base_url)


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings().index_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _hash(policies: list[Policy]) -> str:
    text = "".join(f"{p.doc_id}{p.title}{p.body}" for p in policies) + settings().embed_model
    return hashlib.md5(text.encode()).hexdigest()


async def build_index(force: bool = False) -> int:
    Path(settings().index_path).parent.mkdir(parents=True, exist_ok=True)
    policies = load_policies()
    conn = connect()
    try:
        current = conn.execute("SELECT value FROM meta WHERE key = 'hash'").fetchone()
        if not force and current and current[0] == _hash(policies):
            conn.close()
            return 0
    except sqlite3.OperationalError:
        pass  # индекса нет

    texts, sources = [], []
    for policy in policies:
        for chunk in chunks(policy):
            texts.append(chunk)
            sources.append(policy.doc_id)
    vectors = await embedder().aembed_documents(texts)

    conn.executescript(
        "DROP TABLE IF EXISTS docs; DROP TABLE IF EXISTS pieces;"
        " DROP TABLE IF EXISTS vectors; DROP TABLE IF EXISTS meta;"
        "CREATE TABLE docs (doc_id TEXT PRIMARY KEY, title TEXT, body TEXT);"
        "CREATE TABLE pieces (id INTEGER PRIMARY KEY, doc_id TEXT, text TEXT);"
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"
    )
    conn.execute(f"CREATE VIRTUAL TABLE vectors USING vec0(embedding float[{len(vectors[0])}] distance_metric=cosine)")
    conn.executemany("INSERT INTO docs VALUES (?, ?, ?)", [(p.doc_id, p.title, p.body) for p in policies])
    for i, (doc_id, text, vector) in enumerate(zip(sources, texts, vectors), start=1):
        conn.execute("INSERT INTO pieces VALUES (?, ?, ?)", (i, doc_id, text))
        conn.execute("INSERT INTO vectors(rowid, embedding) VALUES (?, ?)", (i, sqlite_vec.serialize_float32(vector)))
    conn.execute("INSERT INTO meta VALUES ('hash', ?)", (_hash(policies),))
    conn.commit()
    conn.close()
    return len(texts)


async def search(query: str, top_k: int = 3) -> list[dict]:
    vector = await embedder().aembed_query(query)
    conn = connect()
    rows = conn.execute(
        "SELECT p.doc_id, v.distance FROM vectors v JOIN pieces p ON p.id = v.rowid "
        "WHERE v.embedding MATCH ? AND k = 30 ORDER BY v.distance",
        (sqlite_vec.serialize_float32(vector),),
    ).fetchall()

    best: dict[str, float] = {}
    for doc_id, distance in rows:
        best.setdefault(doc_id, distance)
    docs = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT doc_id, title, body FROM docs")}
    conn.close()

    found = sorted(best.items(), key=lambda kv: kv[1])[:top_k]
    return [{"doc_id": doc_id, "title": docs[doc_id][0], "text": docs[doc_id][1],
             "score": round(1 - distance, 3)} for doc_id, distance in found]


if __name__ == "__main__":
    import asyncio

    built = asyncio.run(build_index(force=True))
    print(f"Индекс правил пересобран: {built} кусков в {settings().index_path}")
