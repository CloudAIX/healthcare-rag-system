"""Google BigQuery vector store — the thin GCP variant of the retrieval layer.

Runs in the BigQuery sandbox (no billing account required) using the
VECTOR_SEARCH SQL function in brute-force mode — the honest fit for a
95-chunk corpus, instead of paying for an always-on Vertex vector endpoint.
Vector-only retrieval; the cross-encoder reranker still applies on top
(same as the Azure path's rerank stage).

Duck-types the surfaces the Retriever uses: query() hit dicts and
get_or_create_collection() with .get(ids)/.count().

Auth (either):
  - Application Default Credentials (gcloud auth application-default login)
  - GOOGLE_OAUTH_ACCESS_TOKEN env (e.g. $(gcloud auth print-access-token))
Config: GOOGLE_CLOUD_PROJECT, optional BQ_DATASET (default healthcare_rag).
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from sentence_transformers import SentenceTransformer

EMBED_DIM = 384


def _to_list(vec):
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)


def load_config():
    p = Path(__file__).parent.parent.parent / "config" / "retrieval_config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def _client():
    from google.cloud import bigquery

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT not set")
    token = os.getenv("GOOGLE_OAUTH_ACCESS_TOKEN")
    if token:
        from google.oauth2.credentials import Credentials

        return bigquery.Client(project=project, credentials=Credentials(token=token))
    return bigquery.Client(project=project)


class _CollectionShim:
    def __init__(self, store):
        self._s = store

    def count(self):
        q = f"SELECT COUNT(*) AS n FROM `{self._s.table_ref}`"
        return list(self._s.client.query(q).result())[0]["n"]

    def get(self, ids=None, include=None):
        if not ids:
            return {"ids": [], "documents": [], "metadatas": []}
        from google.cloud import bigquery

        q = f"SELECT * FROM `{self._s.table_ref}` WHERE id IN UNNEST(@ids)"
        job = self._s.client.query(q, job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("ids", "STRING", list(ids))]))
        found, docs, metas = [], [], []
        for r in job.result():
            found.append(r["id"])
            docs.append(r["text"])
            metas.append({
                "chunk_id": r["id"], "document_title": r["document_title"],
                "document_filename": r["document_filename"],
                "page_numbers": r["page_numbers"], "sections": r["sections"],
                "chunk_index": r["chunk_index"],
            })
        return {"ids": found, "documents": docs, "metadatas": metas}


class BigQueryStore:
    def __init__(self, config=None):
        if config is None:
            config = load_config()
        bq = config.get("bigquery", {})
        self.dataset = os.getenv("BQ_DATASET", bq.get("dataset", "healthcare_rag"))
        self.table = bq.get("table", "chunks")
        self.model_name = config["embedding"]["model"]
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self._c = None

    @property
    def client(self):
        if self._c is None:
            self._c = _client()
        return self._c

    @property
    def table_ref(self):
        return f"{self.client.project}.{self.dataset}.{self.table}"

    # ── lifecycle ────────────────────────────────────────────────────────
    def _schema(self):
        from google.cloud import bigquery

        return [
            bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("text", "STRING"),
            bigquery.SchemaField("document_title", "STRING"),
            bigquery.SchemaField("document_filename", "STRING"),
            bigquery.SchemaField("page_numbers", "STRING"),
            bigquery.SchemaField("sections", "STRING"),
            bigquery.SchemaField("chunk_index", "INTEGER"),
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
        ]

    def ensure_table(self):
        from google.cloud import bigquery

        self.client.create_dataset(self.dataset, exists_ok=True)
        self.client.create_table(
            bigquery.Table(self.table_ref, schema=self._schema()), exists_ok=True)
        print(f"BigQuery table ready: {self.table_ref}")

    def reset(self):
        self.client.delete_table(self.table_ref, not_found_ok=True)

    # ── ingest ───────────────────────────────────────────────────────────
    def embed_chunks(self, chunks, batch_size=64):
        rows = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            embs = self.model.encode([c.text for c in batch], show_progress_bar=False)
            for c, e in zip(batch, embs):
                m = c.to_metadata()
                rows.append({
                    "id": c.chunk_id, "text": c.text,
                    "document_title": m.get("document_title", "Unknown"),
                    "document_filename": m.get("document_filename", "unknown.pdf"),
                    "page_numbers": str(m.get("page_numbers", "1")),
                    "sections": str(m.get("sections", "")),
                    "chunk_index": int(m.get("chunk_index", 0)),
                    "embedding": _to_list(e),
                })
        # Batch load job, not streaming insert: streaming is blocked in the
        # BigQuery sandbox, load jobs are free.
        from google.cloud import bigquery

        job = self.client.load_table_from_json(
            rows, self.table_ref,
            job_config=bigquery.LoadJobConfig(
                write_disposition="WRITE_TRUNCATE",
                schema=self._schema()))  # explicit: autodetect re-types "26" as INTEGER
        job.result()
        print(f"BigQuery '{self.table_ref}': loaded {len(rows)} chunks")

    # ── query (VECTOR_SEARCH, brute force — no index needed at this size) ─
    def query(self, query_text, top_k=5):
        from google.cloud import bigquery

        qe = _to_list(self.model.encode([query_text])[0])
        sql = f"""
        SELECT base.id, base.text, base.document_title, base.document_filename,
               base.page_numbers, base.sections, base.chunk_index, distance
        FROM VECTOR_SEARCH(
            TABLE `{self.table_ref}`, 'embedding',
            (SELECT @qe AS embedding), 'embedding',
            top_k => @k, distance_type => 'COSINE')
        """
        job = self.client.query(sql, job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("qe", "FLOAT64", qe),
                bigquery.ScalarQueryParameter("k", "INT64", top_k),
            ]))
        hits = []
        for r in job.result():
            hits.append({
                "id": r["id"], "text": r["text"],
                "metadata": {
                    "chunk_id": r["id"], "document_title": r["document_title"],
                    "document_filename": r["document_filename"],
                    "page_numbers": str(r["page_numbers"]), "sections": str(r["sections"]),
                    "chunk_index": r["chunk_index"],
                },
                "distance": float(r["distance"]),  # cosine distance, lower = better
            })
        return hits

    def get_or_create_collection(self):
        return _CollectionShim(self)
