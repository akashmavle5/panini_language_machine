"""
panini_exporter.py
==================

Panini Language Machine — Symbolic Analysis Exporter

File 7/14.

Purpose
-------
Serialize the outputs of the Marathi/Pāṇinian analysis stack into reusable
machine-readable datasets.

Pipeline:

    MarathiParser
          ↓
    ParsedSentence
          ↓
    PaniniExporter
      ├── JSON
      ├── JSONL
      ├── TSV
      ├── training examples
      ├── rule observations
      └── graph-oriented records

The exporter is deliberately separate from linguistic inference. It does not
change parser decisions; it converts existing symbolic analyses into stable
data contracts.

Design goals
------------
1. Preserve ambiguity.
2. Preserve provenance/evidence.
3. Preserve parser scores.
4. Preserve Paninian feature bundles.
5. Make outputs deterministic.
6. Support corpus-scale batch export.
7. Produce data suitable for later Kāraka, neuro-symbolic training, and LLM
   experiments.
8. Avoid silently inventing labels that the parser did not establish.

Formats
-------
JSON
    Complete structured document.

JSONL
    One sentence/document per line; suitable for large corpora.

TSV
    Flat token-level representation for spreadsheet/SQL workflows.

Training JSONL
    Input/target/metadata records. The target is explicitly symbolic and
    derived from the parser output; it is not presented as ground truth
    beyond what the source analysis supports.

Graph JSONL
    Nodes and observed relations. Dependency/Kāraka relations are left empty
    unless supplied by a later dependency engine.

No external packages are required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from marathi_parser import (
    MarathiParser,
    MorphAnalysis,
    ParsedSentence,
    PaninianFeatureBridge,
)


# ============================================================================
# 1. Serialization helpers
# ============================================================================

def enum_value(value: Any) -> Any:
    """Convert Enum-like values to their serializable representation."""
    if hasattr(value, "value"):
        return value.value

    return value


def stable_json(data: Any) -> str:
    """Deterministic JSON serialization used for IDs and reproducibility."""
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sentence_id(text: str) -> str:
    """
    Stable sentence identifier.

    SHA-256 is used only as a content identifier; it is not a linguistic
    judgment.
    """
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]


def analysis_id(
    sentence_key: str,
    token_index: int,
    candidate_rank: int,
    analysis: MorphAnalysis,
) -> str:
    payload = {
        "sentence_id": sentence_key,
        "token_index": token_index,
        "candidate_rank": candidate_rank,
        "features": analysis.features.to_dict(),
        "source": analysis.source,
    }

    return hashlib.sha256(
        stable_json(payload).encode("utf-8")
    ).hexdigest()[:20]


# ============================================================================
# 2. Export configuration
# ============================================================================

@dataclass(frozen=True)
class ExportOptions:
    """
    Controls output size and ambiguity policy.
    """

    include_traces: bool = True
    include_evidence: bool = True
    include_all_candidates: bool = True
    max_candidates_per_token: Optional[int] = None
    include_unknown_tokens: bool = True
    include_paninian_context: bool = True


# ============================================================================
# 3. Export records
# ============================================================================

@dataclass
class SentenceRecord:
    record_type: str
    sentence_id: str
    language: str
    script: str
    text: str
    token_count: int
    word_count: int
    ambiguous_token_count: int
    unknown_token_count: int
    tokens: List[Dict[str, Any]]
    paninian_context: Optional[Dict[str, Any]]
    traces: Optional[List[Dict[str, Any]]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenRecord:
    record_type: str
    sentence_id: str
    token_index: int
    surface: str
    token_kind: str
    start: int
    end: int
    candidate_rank: int
    candidate_count: int
    analysis_id: str
    lemma: Optional[str]
    pos: str
    gender: str
    number: str
    person: str
    case: str
    tense: str
    aspect: str
    mood: str
    dhatu: Optional[str]
    pratyaya: Optional[str]
    lakara: Optional[str]
    vibhakti_marker: Optional[str]
    karaka_candidate: Optional[str]
    score: float
    source: str
    evidence: List[str]
    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingRecord:
    """
    Symbolic training/evaluation representation.

    `target` is generated from parser analysis, so consumers must distinguish
    parser-derived supervision from externally validated gold labels.
    """

    record_type: str
    sentence_id: str
    input: str
    target: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphRecord:
    """
    Graph-oriented export.

    The parser supplies token nodes. Relations are intentionally empty here;
    File 8 will be able to populate Kāraka/dependency edges.
    """

    record_type: str
    sentence_id: str
    text: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# 4. Core exporter
# ============================================================================

class PaniniExporter:
    """
    Convert ParsedSentence objects into stable export records.
    """

    def __init__(
        self,
        options: Optional[ExportOptions] = None,
    ) -> None:
        self.options = options or ExportOptions()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _feature(
        features: Any,
        name: str,
        default: Any = None,
    ) -> Any:
        value = getattr(features, name, default)
        return enum_value(value)

    def _candidate_slice(
        self,
        candidates: Sequence[MorphAnalysis],
    ) -> Sequence[MorphAnalysis]:
        limit = self.options.max_candidates_per_token

        if limit is None:
            return candidates

        return candidates[: max(0, limit)]

    @staticmethod
    def _word_count(
        parsed: ParsedSentence,
    ) -> int:
        return sum(
            1
            for token in parsed.tokens
            if token.kind.value == "word"
        )

    def _unknown_count(
        self,
        parsed: ParsedSentence,
    ) -> int:
        count = 0

        for token in parsed.tokens:
            if token.kind.value != "word":
                continue

            analysis = parsed.best_analysis(token.index)

            if (
                analysis is not None
                and enum_value(analysis.features.pos) == "unknown"
            ):
                count += 1

        return count

    def _ambiguous_count(
        self,
        parsed: ParsedSentence,
    ) -> int:
        return sum(
            1
            for token in parsed.tokens
            if len(parsed.analyses.get(token.index, [])) > 1
        )

    # ------------------------------------------------------------------
    # Sentence record
    # ------------------------------------------------------------------

    def sentence_record(
        self,
        parsed: ParsedSentence,
    ) -> SentenceRecord:
        sid = sentence_id(parsed.text)

        token_payload = []

        for token in parsed.tokens:
            token_payload.append(
                {
                    "index": token.index,
                    "text": token.text,
                    "kind": token.kind.value,
                    "start": token.start,
                    "end": token.end,
                    "candidate_count": len(
                        parsed.analyses.get(
                            token.index,
                            [],
                        )
                    ),
                }
            )

        context = None

        if self.options.include_paninian_context:
            context = PaninianFeatureBridge.to_context(
                parsed
            )

        traces = None

        if self.options.include_traces:
            traces = [
                trace.to_dict()
                for trace in parsed.traces
            ]

        return SentenceRecord(
            record_type="panini_sentence",
            sentence_id=sid,
            language="mr",
            script="Devanagari",
            text=parsed.text,
            token_count=len(parsed.tokens),
            word_count=self._word_count(parsed),
            ambiguous_token_count=self._ambiguous_count(parsed),
            unknown_token_count=self._unknown_count(parsed),
            tokens=token_payload,
            paninian_context=context,
            traces=traces,
        )

    # ------------------------------------------------------------------
    # Token records
    # ------------------------------------------------------------------

    def token_records(
        self,
        parsed: ParsedSentence,
    ) -> List[TokenRecord]:
        sid = sentence_id(parsed.text)
        result: List[TokenRecord] = []

        for token in parsed.tokens:
            candidates = parsed.analyses.get(
                token.index,
                [],
            )

            if not candidates:
                continue

            selected = self._candidate_slice(
                candidates
            )

            if (
                not self.options.include_all_candidates
                and selected
            ):
                selected = selected[:1]

            if (
                not self.options.include_unknown_tokens
                and selected
                and all(
                    enum_value(
                        candidate.features.pos
                    ) == "unknown"
                    for candidate in selected
                )
            ):
                continue

            for rank, analysis in enumerate(
                selected,
                start=1,
            ):
                features = analysis.features

                result.append(
                    TokenRecord(
                        record_type="panini_token",
                        sentence_id=sid,
                        token_index=token.index,
                        surface=token.text,
                        token_kind=token.kind.value,
                        start=token.start,
                        end=token.end,
                        candidate_rank=rank,
                        candidate_count=len(candidates),
                        analysis_id=analysis_id(
                            sid,
                            token.index,
                            rank,
                            analysis,
                        ),
                        lemma=features.lemma,
                        pos=enum_value(features.pos),
                        gender=enum_value(features.gender),
                        number=enum_value(features.number),
                        person=enum_value(features.person),
                        case=enum_value(features.case),
                        tense=enum_value(features.tense),
                        aspect=enum_value(features.aspect),
                        mood=enum_value(features.mood),
                        dhatu=features.dhatu,
                        pratyaya=features.pratyaya,
                        lakara=features.lakara,
                        vibhakti_marker=(
                            features.vibhakti_marker
                        ),
                        karaka_candidate=(
                            features.karaka_candidate
                        ),
                        score=float(analysis.score),
                        source=analysis.source,
                        evidence=(
                            list(analysis.evidence)
                            if self.options.include_evidence
                            else []
                        ),
                        extra=dict(features.extra),
                    )
                )

        return result

    # ------------------------------------------------------------------
    # Training records
    # ------------------------------------------------------------------

    def training_record(
        self,
        parsed: ParsedSentence,
    ) -> TrainingRecord:
        """
        Build a symbolic supervision example.

        The target contains all exported candidate analyses. This preserves
        ambiguity rather than manufacturing a single "gold" answer.
        """
        sid = sentence_id(parsed.text)

        tokens = []

        for token in parsed.tokens:
            candidates = parsed.analyses.get(
                token.index,
                [],
            )

            candidates = self._candidate_slice(
                candidates
            )

            candidate_payload = []

            for rank, candidate in enumerate(
                candidates,
                start=1,
            ):
                candidate_payload.append(
                    {
                        "rank": rank,
                        "analysis_id": analysis_id(
                            sid,
                            token.index,
                            rank,
                            candidate,
                        ),
                        "score": candidate.score,
                        "source": candidate.source,
                        "features": candidate.features.to_dict(),
                        "evidence": (
                            list(candidate.evidence)
                            if self.options.include_evidence
                            else []
                        ),
                    }
                )

            tokens.append(
                {
                    "token_index": token.index,
                    "surface": token.text,
                    "kind": token.kind.value,
                    "candidates": candidate_payload,
                }
            )

        target = {
            "representation": "paninian_morphology",
            "tokens": tokens,
        }

        metadata = {
            "language": "mr",
            "script": "Devanagari",
            "supervision_type": "parser_derived",
            "gold_status": "not_validated_gold",
            "ambiguity_preserved": True,
        }

        return TrainingRecord(
            record_type="panini_training_example",
            sentence_id=sid,
            input=parsed.text,
            target=target,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Graph record
    # ------------------------------------------------------------------

    def graph_record(
        self,
        parsed: ParsedSentence,
    ) -> GraphRecord:
        sid = sentence_id(parsed.text)

        nodes: List[Dict[str, Any]] = []

        for token in parsed.tokens:
            best = parsed.best_analysis(
                token.index
            )

            node = {
                "id": f"{sid}:t{token.index}",
                "token_index": token.index,
                "surface": token.text,
                "kind": token.kind.value,
                "features": (
                    best.features.to_dict()
                    if best is not None
                    else {}
                ),
            }

            nodes.append(node)

        return GraphRecord(
            record_type="panini_graph",
            sentence_id=sid,
            text=parsed.text,
            nodes=nodes,
            edges=[],
            metadata={
                "language": "mr",
                "script": "Devanagari",
                "edge_status": "not_inferred",
                "next_layer": "karaka_dependency",
            },
        )

    # ------------------------------------------------------------------
    # Full document
    # ------------------------------------------------------------------

    def export_document(
        self,
        parsed: ParsedSentence,
    ) -> Dict[str, Any]:
        return {
            "schema": "panini-language-machine/v1",
            "record_type": "panini_document",
            "sentence": self.sentence_record(
                parsed
            ).to_dict(),
            "tokens": [
                record.to_dict()
                for record in self.token_records(parsed)
            ],
            "training": self.training_record(
                parsed
            ).to_dict(),
            "graph": self.graph_record(
                parsed
            ).to_dict(),
        }

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def export_many(
        self,
        parsed_sentences: Iterable[ParsedSentence],
    ) -> Iterator[Dict[str, Any]]:
        for parsed in parsed_sentences:
            yield self.export_document(parsed)


# ============================================================================
# 5. File writers
# ============================================================================

class PaniniExportWriter:
    """
    Output writer for JSON, JSONL and TSV.
    """

    @staticmethod
    def write_json(
        payload: Any,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return destination

    @staticmethod
    def write_jsonl(
        records: Iterable[Mapping[str, Any]],
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        with destination.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        return destination

    @staticmethod
    def write_token_tsv(
        records: Iterable[TokenRecord],
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        fieldnames = [
            "record_type",
            "sentence_id",
            "token_index",
            "surface",
            "token_kind",
            "start",
            "end",
            "candidate_rank",
            "candidate_count",
            "analysis_id",
            "lemma",
            "pos",
            "gender",
            "number",
            "person",
            "case",
            "tense",
            "aspect",
            "mood",
            "dhatu",
            "pratyaya",
            "lakara",
            "vibhakti_marker",
            "karaka_candidate",
            "score",
            "source",
            "evidence",
            "extra",
        ]

        with destination.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                delimiter="\t",
            )

            writer.writeheader()

            for record in records:
                row = record.to_dict()

                row["evidence"] = json.dumps(
                    row["evidence"],
                    ensure_ascii=False,
                )

                row["extra"] = json.dumps(
                    row["extra"],
                    ensure_ascii=False,
                )

                writer.writerow(row)

        return destination


# ============================================================================
# 6. Corpus exporter
# ============================================================================

class MarathiCorpusExporter:
    """
    Convenience class that owns the parser + exporter and supports large
    line-oriented corpora without loading all source sentences into memory.
    """

    def __init__(
        self,
        parser: Optional[MarathiParser] = None,
        options: Optional[ExportOptions] = None,
    ) -> None:
        self.parser = parser or MarathiParser()
        self.exporter = PaniniExporter(
            options=options
        )

    def parse_lines(
        self,
        lines: Iterable[str],
    ) -> Iterator[ParsedSentence]:
        for line in lines:
            text = line.strip()

            if not text:
                continue

            yield self.parser.parse(text)

    def export_jsonl(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> int:
        count = 0

        def records() -> Iterator[Dict[str, Any]]:
            nonlocal count

            with Path(input_path).open(
                "r",
                encoding="utf-8",
            ) as handle:
                for parsed in self.parse_lines(handle):
                    count += 1
                    yield self.exporter.export_document(
                        parsed
                    )

        PaniniExportWriter.write_jsonl(
            records(),
            output_path,
        )

        return count

    def export_training_jsonl(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> int:
        count = 0

        def records() -> Iterator[Dict[str, Any]]:
            nonlocal count

            with Path(input_path).open(
                "r",
                encoding="utf-8",
            ) as handle:
                for parsed in self.parse_lines(handle):
                    count += 1

                    yield (
                        self.exporter
                        .training_record(parsed)
                        .to_dict()
                    )

        PaniniExportWriter.write_jsonl(
            records(),
            output_path,
        )

        return count

    def export_graph_jsonl(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> int:
        count = 0

        def records() -> Iterator[Dict[str, Any]]:
            nonlocal count

            with Path(input_path).open(
                "r",
                encoding="utf-8",
            ) as handle:
                for parsed in self.parse_lines(handle):
                    count += 1

                    yield (
                        self.exporter
                        .graph_record(parsed)
                        .to_dict()
                    )

        PaniniExportWriter.write_jsonl(
            records(),
            output_path,
        )

        return count


# ============================================================================
# 7. Validation
# ============================================================================

def validate_export_document(
    document: Mapping[str, Any],
) -> List[str]:
    """
    Structural validation only.

    It does not validate linguistic correctness.
    """
    errors: List[str] = []

    required = {
        "schema",
        "record_type",
        "sentence",
        "tokens",
        "training",
        "graph",
    }

    missing = required.difference(
        document.keys()
    )

    for key in sorted(missing):
        errors.append(
            f"missing top-level field: {key}"
        )

    sentence = document.get("sentence")

    if not isinstance(sentence, Mapping):
        errors.append(
            "sentence must be an object"
        )
    else:
        for key in (
            "sentence_id",
            "language",
            "script",
            "text",
            "token_count",
        ):
            if key not in sentence:
                errors.append(
                    f"missing sentence field: {key}"
                )

    tokens = document.get("tokens")

    if not isinstance(tokens, list):
        errors.append(
            "tokens must be a list"
        )

    training = document.get("training")

    if not isinstance(training, Mapping):
        errors.append(
            "training must be an object"
        )

    graph = document.get("graph")

    if not isinstance(graph, Mapping):
        errors.append(
            "graph must be an object"
        )

    return errors


# ============================================================================
# 8. CLI
# ============================================================================

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="panini_exporter",
        description=(
            "Export Marathi/Pāṇinian parser analyses "
            "to JSON, JSONL, TSV and training formats."
        ),
    )

    parser.add_argument(
        "text",
        nargs="?",
        help="Single Marathi sentence.",
    )

    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Export a single sentence as complete JSON.",
    )

    parser.add_argument(
        "--training",
        metavar="PATH",
        help="Export a single sentence as training JSON.",
    )

    parser.add_argument(
        "--graph",
        metavar="PATH",
        help="Export a single sentence as graph JSON.",
    )

    parser.add_argument(
        "--tokens-tsv",
        metavar="PATH",
        help="Export single-sentence token analyses as TSV.",
    )

    parser.add_argument(
        "--input",
        metavar="PATH",
        help="UTF-8 text file, one sentence per line.",
    )

    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Output path used with --input.",
    )

    parser.add_argument(
        "--format",
        choices=[
            "jsonl",
            "training-jsonl",
            "graph-jsonl",
        ],
        default="jsonl",
        help="Batch output format.",
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Maximum candidate analyses exported per token.",
    )

    return parser


def main(
    argv: Optional[Sequence[str]] = None,
) -> int:
    args = build_cli().parse_args(argv)

    options = ExportOptions(
        max_candidates_per_token=args.max_candidates
    )

    parser = MarathiParser()
    exporter = PaniniExporter(options)

    if args.input:
        if not args.output:
            print(
                "--output is required with --input",
                file=sys.stderr,
            )
            return 2

        corpus = MarathiCorpusExporter(
            parser=parser,
            options=options,
        )

        if args.format == "jsonl":
            count = corpus.export_jsonl(
                args.input,
                args.output,
            )
        elif args.format == "training-jsonl":
            count = corpus.export_training_jsonl(
                args.input,
                args.output,
            )
        else:
            count = corpus.export_graph_jsonl(
                args.input,
                args.output,
            )

        print(
            f"Exported {count} sentences to {args.output}"
        )

        return 0

    if not args.text:
        print(
            "Provide a Marathi sentence or --input.",
            file=sys.stderr,
        )
        return 2

    parsed = parser.parse(args.text)
    document = exporter.export_document(parsed)

    if args.json:
        PaniniExportWriter.write_json(
            document,
            args.json,
        )
        print(f"JSON written to {args.json}")

    elif args.training:
        record = exporter.training_record(
            parsed
        ).to_dict()

        PaniniExportWriter.write_json(
            record,
            args.training,
        )
        print(
            f"Training JSON written to {args.training}"
        )

    elif args.graph:
        record = exporter.graph_record(
            parsed
        ).to_dict()

        PaniniExportWriter.write_json(
            record,
            args.graph,
        )
        print(
            f"Graph JSON written to {args.graph}"
        )

    elif args.tokens_tsv:
        records = exporter.token_records(parsed)

        PaniniExportWriter.write_token_tsv(
            records,
            args.tokens_tsv,
        )

        print(
            f"Token TSV written to {args.tokens_tsv}"
        )

    else:
        print(
            json.dumps(
                document,
                ensure_ascii=False,
                indent=2,
            )
        )

    return 0


# ============================================================================
# 9. Self-test
# ============================================================================

def self_test() -> None:
    parser = MarathiParser()

    parsed = parser.parse(
        "राम पुस्तक घेऊन गेला."
    )

    exporter = PaniniExporter()

    # Sentence record.
    sentence = exporter.sentence_record(
        parsed
    )

    assert sentence.record_type == "panini_sentence"
    assert sentence.language == "mr"
    assert sentence.script == "Devanagari"
    assert sentence.text == "राम पुस्तक घेऊन गेला."
    assert sentence.token_count == 5

    # Token records.
    token_records = exporter.token_records(
        parsed
    )

    assert token_records
    assert token_records[0].surface == "राम"
    assert token_records[0].pos == "noun"

    # Candidate provenance.
    assert token_records[0].source == "lexicon"
    assert token_records[0].evidence

    # Stable IDs.
    first_id = token_records[0].analysis_id

    second_export = exporter.token_records(
        parsed
    )

    assert second_export[0].analysis_id == first_id

    # Ambiguity preserved.
    ambiguous = exporter.token_records(
        parser.parse("गेला.")
    )

    assert len(ambiguous) >= 1

    # Training record.
    training = exporter.training_record(
        parsed
    )

    assert (
        training.metadata["supervision_type"]
        == "parser_derived"
    )

    assert (
        training.metadata["gold_status"]
        == "not_validated_gold"
    )

    # Graph record.
    graph = exporter.graph_record(
        parsed
    )

    assert len(graph.nodes) == 5
    assert graph.edges == []

    # Full document.
    document = exporter.export_document(
        parsed
    )

    errors = validate_export_document(
        document
    )

    assert errors == []

    assert (
        document["schema"]
        == "panini-language-machine/v1"
    )

    # JSON serialization.
    payload = json.dumps(
        document,
        ensure_ascii=False,
    )

    assert "राम" in payload
    assert "panini_token" in payload

    # Writer tests.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        json_path = PaniniExportWriter.write_json(
            document,
            root / "document.json",
        )

        assert json_path.exists()

        jsonl_path = PaniniExportWriter.write_jsonl(
            [document],
            root / "document.jsonl",
        )

        assert jsonl_path.exists()

        tsv_path = PaniniExportWriter.write_token_tsv(
            token_records,
            root / "tokens.tsv",
        )

        assert tsv_path.exists()

        lines = tsv_path.read_text(
            encoding="utf-8"
        ).splitlines()

        assert len(lines) >= 2
        assert "surface" in lines[0]

    # Corpus exporter.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        source = root / "sentences.txt"
        output = root / "corpus.jsonl"

        source.write_text(
            "राम आहे.\nसीता गेली.\n",
            encoding="utf-8",
        )

        corpus = MarathiCorpusExporter()

        count = corpus.export_jsonl(
            source,
            output,
        )

        assert count == 2
        assert output.exists()

        exported_lines = [
            line
            for line in output.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        assert len(exported_lines) == 2

    # Feature bridge must remain intact.
    assert (
        document["sentence"]["paninian_context"]
        is not None
    )


# ============================================================================
# 10. Demo
# ============================================================================

def demo() -> None:
    sentence = "राम पुस्तक घेऊन गेला."

    parser = MarathiParser()
    exporter = PaniniExporter()

    parsed = parser.parse(sentence)
    document = exporter.export_document(parsed)

    print("=" * 96)
    print("PANINI LANGUAGE MACHINE — FILE 7 DEMO")
    print("=" * 96)

    print("\nSentence:")
    print(sentence)

    print("\nExport schema:")
    print(document["schema"])

    print("\nSentence ID:")
    print(document["sentence"]["sentence_id"])

    print("\nToken records:")
    for record in document["tokens"]:
        print(
            f"  [{record['token_index']}] "
            f"{record['surface']} "
            f"POS={record['pos']} "
            f"lemma={record['lemma']} "
            f"score={record['score']:.2f} "
            f"source={record['source']}"
        )

    print("\nGraph:")
    print(
        json.dumps(
            document["graph"],
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nSelf-test:")
    self_test()
    print("PASS")


if __name__ == "__main__":
    main()
