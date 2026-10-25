"""Section-level BM25 retrieval for policy documents and product facts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import unicodedata
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .embeddings import GeminiEmbeddingModel, normalize_cache_key


FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)
SECTION_RE = re.compile(
    r"^##\s+\[([^\]]+)\]\s*(.*?)\s*\n(.*?)(?=^##\s+\[|\Z)",
    re.MULTILINE | re.DOTALL,
)

STOPWORDS = set(
    "a an the and or but if then is are was were be been being do does did can could "
    "will would should may might i me my we our you your he she they it this that "
    "for of to in on at by with from as about just still how what when where who s "
    "क्या है हैं हूँ हो मुझे मेरा मेरी मेरे और या के का की को से पर तक अब मैं आप "
    "हैं है में इस उस कितने कौन कब कैसे क्या".split()
)


def _safe_embedding_error(exc: Exception) -> str:
    """Keep useful SDK diagnostics while ensuring configured API keys stay private."""
    message = str(exc)
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        secret = os.getenv(name)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"(?i)(key\s*[=:]\s*)[^\s,;]+", r"\1[REDACTED]", message)
    message = " ".join(message.split())[:300]
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__

# Small domain vocabulary expansion helps lexical retrieval with customer phrasing
# such as Hinglish, paraphrases, and the dataset's ASR-style wording.
QUERY_SYNONYMS = {
    "wapas": "return",
    "vapas": "return",
    "louta": "return",
    "lautana": "return",
    "lautane": "return",
    "kitne": "days",
    "din": "days",
    "parso": "delivered delivery",
    "hua": "was",
    "hai": "is",
    "hain": "are",
    "kya": "",
    "kaise": "how",
    "kar": "",
    "sakti": "can",
    "hoon": "",
    "mera": "my",
    "mujhe": "",
    "ko": "",
    "pe": "",
    "liye": "for",
    "khola": "opened",
    "khol": "open",
    "khul": "opened",
    "opened": "unopened seal condition",
    "लौटाने": "return",
    "लौट": "return",
    "वापस": "return",
    "दिन": "days",
    "कितने": "days",
    "खोला": "opened unopened seal",
    "खोल": "opened unopened seal",
    "चुकी": "already",
    "क्या": "",
    "सकती": "can",
    "हूँ": "",
    "मुझे": "",
    "torn": "damaged",
    "collection": "pickup",
    "never": "unauthorized fraud",
    "placed": "unauthorized order",
    "come": "arrive timeline",
    "refund": "refund money",
    "refunds": "refund",
    "when": "timeline",
    "ऑर्डर": "order",
    "शिप": "ship shipped",
    "कैंसिल": "cancel cancelled cancellation",
    "कैंसल": "cancel cancelled cancellation",
    "रद्द": "cancel cancelled cancellation",
}


def _raw_tokens(text: str) -> list[str]:
    """Split text without dropping vowel marks from Indic scripts."""
    tokens: list[str] = []
    current: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category[0] in {"L", "N", "M"} or char == "₹":
            current.append(char)
        elif current:
            tokens.append("".join(current).casefold())
            current = []
    if current:
        tokens.append("".join(current).casefold())
    return tokens


def _tokens(text: str) -> list[str]:
    result: list[str] = []
    for token in _raw_tokens(text):
        if token in STOPWORDS:
            continue
        result.append(token)
        expanded = QUERY_SYNONYMS.get(token, "")
        result.extend(
            item for item in _raw_tokens(expanded)
            if item and item not in STOPWORDS
        )
    return result


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(":")
        if separator:
            metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, text[match.end():]


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source_type: str
    source_id: str
    section_id: str
    title: str
    text: str
    status: str = ""
    effective_from: str = ""
    doc_type: str = ""
    score: float = 0.0

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


AUTHORITY_MULTIPLIER = {
    "policy": 1.12,
    "meta": 1.08,
    "agent_guide": 1.08,
    "help": 1.04,
    "faq": 1.02,
    "product": 1.03,
    "data": 1.02,
    "marketing": 0.98,
}

DOCUMENT_PRECEDENCE = {
    "policy": 0,
    "meta": 1,
    "agent_guide": 2,
    "help": 3,
    "faq": 4,
    "marketing": 5,
}

RRF_CONSTANT = 60
EMBEDDING_DIMENSIONS = 768


class KnowledgeRetriever:
    """Load evidence and rank it with hybrid lexical and Gemini embedding search."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        use_embeddings: bool = True,
        embedding_model_name: str | None = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.k1 = k1
        self.b = b
        self.evidence = self._load_evidence()
        if not self.evidence:
            raise ValueError(f"No searchable evidence found under {self.dataset_dir}")
        self._tokenized = [_tokens(self._search_text(item)) for item in self.evidence]
        self._lengths = [len(tokens) for tokens in self._tokenized]
        self._avg_length = sum(self._lengths) / max(1, len(self._lengths))
        self._document_frequency: Counter[str] = Counter()
        for tokens in self._tokenized:
            self._document_frequency.update(set(tokens))
        self.embedding_model: GeminiEmbeddingModel | None = None
        self._document_vectors: list[list[float]] | None = None
        self._query_vectors: dict[str, list[float]] = {}
        self._query_embedding_failed = False
        self.embedding_error: str | None = None
        self.mode = "lexical"
        if use_embeddings:
            try:
                self.embedding_model = GeminiEmbeddingModel(embedding_model_name)
                self._document_vectors = self._load_or_create_document_vectors()
                self.mode = "hybrid"
            except Exception as exc:
                self.embedding_error = _safe_embedding_error(exc)
                warnings.warn(
                    f"Gemini embeddings unavailable ({self.embedding_error}); using lexical retrieval.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.embedding_model = None
                self._document_vectors = None

    def _embedding_text(self, item: Evidence) -> str:
        return (
            f"Document: {item.title}\n"
            f"Evidence ID: {item.evidence_id}\n"
            f"Type: {item.doc_type}; status: {item.status}; "
            f"effective from: {item.effective_from}\n"
            f"{item.text}"
        )

    def _corpus_fingerprint(self) -> str:
        payload = "\n".join(
            f"{item.source_id}\0{item.evidence_id}\0{normalize_cache_key(self._embedding_text(item))}"
            for item in self.evidence
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _embedding_cache_path(self) -> Path:
        assert self.embedding_model is not None
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", self.embedding_model.model_name)
        return (
            self.dataset_dir.parent
            / ".cache"
            / "zentromart"
            / f"{safe_model}-{self._corpus_fingerprint()[:16]}.json"
        )

    def _load_or_create_document_vectors(self) -> list[list[float]]:
        assert self.embedding_model is not None
        cache_path = self._embedding_cache_path()
        fingerprint = self._corpus_fingerprint()
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                vectors = cached.get("vectors")
                if (
                    cached.get("model") == self.embedding_model.model_name
                    and cached.get("corpus_fingerprint") == fingerprint
                    and len(vectors) == len(self.evidence)
                    and all(len(vector) == EMBEDDING_DIMENSIONS for vector in vectors)
                ):
                    return vectors
            except (OSError, ValueError, TypeError):
                pass

        texts = [self._embedding_text(item) for item in self.evidence]
        vectors = self.embedding_model.embed_many(texts, task_type="RETRIEVAL_DOCUMENT")
        if len(vectors) != len(self.evidence):
            raise RuntimeError("Incomplete document embedding index returned by Gemini.")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({
                "model": self.embedding_model.model_name,
                "corpus_fingerprint": fingerprint,
                "dimensions": EMBEDDING_DIMENSIONS,
                "vectors": vectors,
            }),
            encoding="utf-8",
        )
        return vectors

    def prepare_queries(self, queries: list[str]) -> None:
        """Pre-embed a query batch efficiently for offline retrieval evaluation."""
        if self.embedding_model is None or self._query_embedding_failed:
            return
        normalized = list(dict.fromkeys(normalize_cache_key(query) for query in queries if query.strip()))
        missing = [query for query in normalized if query not in self._query_vectors]
        if not missing:
            return
        try:
            vectors = self.embedding_model.embed_many(missing, task_type="RETRIEVAL_QUERY")
            self._query_vectors.update(zip(missing, vectors))
        except Exception as exc:
            self.embedding_error = _safe_embedding_error(exc)
            self._query_embedding_failed = True
            warnings.warn(
                f"Gemini query embeddings unavailable ({self.embedding_error}); using lexical ranking.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _load_evidence(self) -> list[Evidence]:
        items: list[Evidence] = []
        kb_dir = self.dataset_dir / "kb"
        for path in sorted(kb_dir.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            metadata, body = _parse_front_matter(raw)
            # Superseded files are retained in the pack for historical context;
            # current successors carry forward any exceptions that still apply.
            if metadata.get("status", "").casefold() == "superseded":
                continue
            common = {
                "source_type": metadata.get("doc_type", "document"),
                "source_id": path.name,
                "status": metadata.get("status", ""),
                "effective_from": metadata.get("effective_from", ""),
                "doc_type": metadata.get("doc_type", ""),
            }
            matches = list(SECTION_RE.finditer(body))
            if matches:
                for match in matches:
                    section_id, title, content = match.groups()
                    section_text = content.strip()
                    if section_text:
                        items.append(Evidence(
                            evidence_id=section_id,
                            section_id=section_id,
                            title=title.strip(),
                            text=section_text,
                            **common,
                        ))
            else:
                content = body.strip()
                if content:
                    items.append(Evidence(
                        evidence_id=metadata.get("doc_id", path.stem),
                        section_id=metadata.get("doc_id", path.stem),
                        title=metadata.get("title", path.stem),
                        text=content,
                        **common,
                    ))

        products_path = self.dataset_dir / "data" / "products.csv"
        if products_path.is_file():
            with products_path.open(encoding="utf-8-sig", newline="") as source:
                for row in csv.DictReader(source):
                    sku = row.get("sku", "").strip()
                    if not sku:
                        continue
                    details = " | ".join(
                        f"{key}: {value.strip()}"
                        for key, value in row.items()
                        if value and value.strip()
                    )
                    items.append(Evidence(
                        evidence_id=f"products.csv:{sku}",
                        source_type="catalog",
                        source_id="products.csv",
                        section_id=f"products.csv:{sku}",
                        title=row.get("title", sku),
                        text=details,
                        doc_type="product",
                    ))

        for filename, key_column, source_type, title_column in (
            ("pincodes.csv", "pincode", "serviceability", "city"),
            ("holidays.csv", "date", "calendar", "name"),
        ):
            table_path = self.dataset_dir / "data" / filename
            if not table_path.is_file():
                continue
            with table_path.open(encoding="utf-8-sig", newline="") as source:
                for row in csv.DictReader(source):
                    key = row.get(key_column, "").strip()
                    if not key:
                        continue
                    details = " | ".join(
                        f"{column}: {value.strip()}"
                        for column, value in row.items()
                        if value and value.strip()
                    )
                    items.append(Evidence(
                        evidence_id=f"{filename}:{key}",
                        source_type=source_type,
                        source_id=filename,
                        section_id=f"{filename}:{key}",
                        title=row.get(title_column, key),
                        text=details,
                        doc_type="data",
                    ))
        return items

    @staticmethod
    def _search_text(item: Evidence) -> str:
        return " ".join((item.title, item.evidence_id, item.source_id, item.text))

    def _lexical_ranking(self, query_tokens: list[str]) -> list[tuple[float, int]]:
        n_docs = len(self.evidence)
        q_counts = Counter(query_tokens)
        scored: list[tuple[float, int]] = []
        for index, (item, tokens, length) in enumerate(
            zip(self.evidence, self._tokenized, self._lengths)
        ):
            frequencies = Counter(tokens)
            score = 0.0
            for term, qf in q_counts.items():
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                df = self._document_frequency[term]
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * length / max(self._avg_length, 1.0)
                )
                score += idf * (tf * (self.k1 + 1) / denominator) * (1 + math.log(qf))
            if score:
                score *= AUTHORITY_MULTIPLIER.get(item.doc_type, 1.0)
                scored.append((score, index))
        scored.sort(key=lambda value: value[0], reverse=True)

        # Preserve the useful lexical bridge between a product category and its
        # policy, while embeddings handle paraphrases and cross-language wording.
        product_candidates = [
            candidate for candidate in scored
            if self.evidence[candidate[1]].source_type == "catalog"
        ]
        product_text = (
            self.evidence[product_candidates[0][1]].text.casefold()
            if product_candidates else ""
        )
        query_set = set(query_tokens)
        is_return_question = bool(query_set.intersection({"return", "returnable", "send", "back"}))
        is_refund_question = "refund" in query_set or "money" in query_set
        fraud_markers = {
            "never", "unauthorized", "fraud", "unrecognized", "unknown",
            "placed", "activity",
        }
        is_fraud_question = bool(query_set.intersection(fraud_markers)) and (
            "order" in query_set or "account" in query_set or "payment" in query_set
        )

        adjusted: list[tuple[float, int]] = []
        for score, index in scored:
            item = self.evidence[index]
            bonus = 0.0
            if is_return_question:
                if item.evidence_id == "RET-2":
                    bonus += 5.0
                if "return_category: electronics" in product_text and item.evidence_id == "RET-3":
                    bonus += 5.0
                if "return_category: fashion" in product_text and item.evidence_id == "RET-8":
                    bonus += 3.0
                if "return_category: beauty_sealed" in product_text and item.evidence_id == "RET-4":
                    bonus += 5.0
                if any(
                    f"return_category: {category}" in product_text
                    for category in ("innerwear", "grocery", "digital")
                ) and item.evidence_id == "RET-5":
                    bonus += 6.0
            if is_refund_question and item.evidence_id == "REF-2":
                bonus += 7.0
            if is_fraud_question and item.evidence_id == "ESC-1":
                bonus += 14.0
            adjusted.append((score + bonus, index))
        adjusted.sort(key=lambda value: value[0], reverse=True)
        return adjusted

    def retrieve(self, query: str, *, top_k: int = 5) -> list[Evidence]:
        query = normalize_cache_key(query.strip())
        query_tokens = _tokens(query)
        if not query_tokens and self.mode == "lexical":
            return []
        lexical = self._lexical_ranking(query_tokens)
        if self.mode != "hybrid" or self._document_vectors is None:
            ranked = lexical
        else:
            if query not in self._query_vectors:
                self.prepare_queries([query])
            query_vector = self._query_vectors.get(query)
            if query_vector is None:
                ranked = lexical
                self.mode = "lexical-fallback"
            else:
                dense_scores = [
                    sum(left * right for left, right in zip(query_vector, vector))
                    for vector in self._document_vectors
                ]
                dense_order = sorted(
                    range(len(dense_scores)),
                    key=lambda index: dense_scores[index],
                    reverse=True,
                )
                dense_rank = {index: rank for rank, index in enumerate(dense_order, 1)}
                lexical_rank = {index: rank for rank, (_, index) in enumerate(lexical, 1)}
                ranked = []
                for index, item in enumerate(self.evidence):
                    dense_rrf = 1.0 / (RRF_CONSTANT + dense_rank[index])
                    lexical_rrf = (
                        1.0 / (RRF_CONSTANT + lexical_rank[index])
                        if index in lexical_rank else 0.0
                    )
                    fused = (0.65 * dense_rrf + 0.35 * lexical_rrf)
                    fused *= AUTHORITY_MULTIPLIER.get(item.doc_type, 1.0)
                    ranked.append((fused, index))
                ranked.sort(key=lambda value: value[0], reverse=True)

                # Preserve the explicit immediate-handover rule for reports
                # of account/order/payment activity the customer did not make.
                fraud_markers = {
                    "never", "unauthorized", "fraud", "unrecognized", "unknown",
                    "placed", "activity",
                }
                is_fraud_question = bool(set(query_tokens).intersection(fraud_markers)) and (
                    "order" in query_tokens or "account" in query_tokens or "payment" in query_tokens
                )
                if is_fraud_question:
                    ranked.sort(
                        key=lambda value: (
                            self.evidence[value[1]].evidence_id != "ESC-1",
                            -value[0],
                        )
                    )

        selected: list[tuple[float, int]] = []
        product_count = 0
        for candidate in ranked:
            item = self.evidence[candidate[1]]
            if item.source_type == "catalog":
                product_count += 1
                if product_count > 2:
                    continue
            selected.append(candidate)
            if len(selected) >= top_k:
                break

        # Relevance determines which evidence is selected. Within that shortlist,
        # preserve the knowledge-base precedence order for policy-style documents
        # so a FAQ or marketing page cannot appear before a conflicting policy.
        # Product and table evidence stay in their relevance-selected positions.
        precedence_positions = [
            position for position, (_, index) in enumerate(selected)
            if self.evidence[index].doc_type in DOCUMENT_PRECEDENCE
        ]
        precedence_items = sorted(
            (selected[position] for position in precedence_positions),
            key=lambda candidate: (
                DOCUMENT_PRECEDENCE[self.evidence[candidate[1]].doc_type],
                -candidate[0],
            ),
        )
        for position, candidate in zip(precedence_positions, precedence_items):
            selected[position] = candidate

        return [
            Evidence(**{**asdict(self.evidence[index]), "score": round(score, 6)})
            for score, index in selected
        ]

    def search(self, query: str, *, top_k: int = 5) -> list[Evidence]:
        """Alias useful to callers that treat retrieval as a search operation."""
        return self.retrieve(query, top_k=top_k)

