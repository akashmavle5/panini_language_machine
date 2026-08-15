"""
karaka_dependency.py
====================

Panini Language Machine — Kāraka / Dependency Engine

File 8/14.

This file preserves the original Kāraka dependency / attention-bias concept:

    Vibhakti / inflection
            ↓
       Kāraka slot
            ↓
       Verb valency (Ākāṅkṣā)
            ↓
    structural dependency
            ↓
    attention-bias matrix

It additionally provides a clean programmatic API for the Marathi parser
built in Files 5–7.  The Sanskrit-style demonstration remains available so
the original structural-attention experiment can be reproduced exactly.

Important:
The +5.0 matrix value is an experimental architectural prior, not a claim
that a particular value is optimal.  It represents a strong symbolic bias
that can later be learned, calibrated, or ablated in File 9.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from marathi_parser import MarathiParser, ParsedSentence
except ImportError:  # allows the Sanskrit demonstration to run independently
    MarathiParser = None
    ParsedSentence = Any


# ============================================================================
# 1. Data structures
# ============================================================================

@dataclass
class KarakaAssignment:
    role: str
    token: str
    index: int
    root: Optional[str] = None
    score: float = 1.0
    evidence: str = "lexicon/vibhakti"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DependencyBinding:
    source_index: int
    target_index: int
    relation: str
    score: float
    evidence: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DependencyAnalysis:
    tokens: List[str]
    verb_index: Optional[int]
    verb: Optional[str]
    assignments: List[KarakaAssignment]
    bindings: List[DependencyBinding]
    missing_required_roles: List[str]
    attention_bias: List[List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_type": "panini_karaka_dependency",
            "tokens": self.tokens,
            "verb_index": self.verb_index,
            "verb": self.verb,
            "assignments": [a.to_dict() for a in self.assignments],
            "bindings": [b.to_dict() for b in self.bindings],
            "missing_required_roles": self.missing_required_roles,
            "attention_bias": self.attention_bias,
        }


# ============================================================================
# 2. Core Kāraka matrix
# ============================================================================

class KarakaDependencyMatrix:
    """
    Symbolic Kāraka resolver and structural attention-bias generator.

    The vocabulary and valency registry intentionally remain explicit and
    inspectable. This makes the symbolic prior easy to compare against a
    purely learned attention mechanism.
    """

    def __init__(self, attention_boost: float = 5.0):
        self.attention_boost = float(attention_boost)

        # Base vocabulary registry from the original File-8 concept.
        self.vocabulary = [
            "hastena",
            "grāmam",
            "rāmāḥ",
            "gacchati",
        ]

        # Surface inflection → semantic Kāraka.
        self.inflection_to_karaka = {
            "hastena": {
                "root": "hasta",
                "role": "Karaṇa",
            },
            "grāmam": {
                "root": "grāma",
                "role": "Karma",
            },
            "rāmāḥ": {
                "root": "rāma",
                "role": "Kartā",
            },
        }

        # Action valency / Ākāṅkṣā.
        self.verb_valency = {
            "gacchati": {
                "root": "gam",
                "demanded_roles": ["Kartā", "Karma"],
                "optional_roles": ["Karaṇa"],
            }
        }

    # ------------------------------------------------------------------
    # Structural extraction
    # ------------------------------------------------------------------

    def extract_structural_roles(
        self,
        token_sequence: Sequence[str],
    ) -> Tuple[Dict[str, Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Parse surface tokens into structural Kāraka assignments.

        Physical sequence is not used to decide the role. The explicit
        inflection registry is the primary signal.
        """
        active_roles: Dict[str, Dict[str, Any]] = {}
        verb_token: Optional[Dict[str, Any]] = None

        for idx, token in enumerate(token_sequence):
            if token in self.inflection_to_karaka:
                meta = self.inflection_to_karaka[token]
                active_roles[meta["role"]] = {
                    "token": token,
                    "index": idx,
                    "root": meta["root"],
                }

            if token in self.verb_valency:
                verb_token = {
                    "token": token,
                    "index": idx,
                }

        return active_roles, verb_token

    # ------------------------------------------------------------------
    # Attention bias
    # ------------------------------------------------------------------

    def compute_paninian_attention_bias(
        self,
        token_sequence: Sequence[str],
    ) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]], List[str]]:
        """
        Generate a structural attention bias matrix.

        A valid Kāraka ↔ verb binding receives `attention_boost` in both
        directions. No bias is introduced for unsupported role/verb pairs.
        """
        seq_len = len(token_sequence)
        bias_matrix = np.zeros(
            (seq_len, seq_len),
            dtype=np.float32,
        )

        active_roles, verb_info = self.extract_structural_roles(
            token_sequence
        )

        if not verb_info:
            return bias_matrix, active_roles, []

        verb_idx = verb_info["index"]
        verb_name = verb_info["token"]
        valency = self.verb_valency[verb_name]

        fulfilled_roles: List[str] = []

        for role, data in active_roles.items():
            noun_idx = data["index"]

            if (
                role in valency["demanded_roles"]
                or role in valency["optional_roles"]
            ):
                bias_matrix[noun_idx, verb_idx] = self.attention_boost
                bias_matrix[verb_idx, noun_idx] = self.attention_boost
                fulfilled_roles.append(role)

        # Explicitly keep diagonal neutral: this matrix is a relational prior,
        # not a self-attention replacement.
        np.fill_diagonal(bias_matrix, 0.0)

        unfulfilled = [
            role
            for role in valency["demanded_roles"]
            if role not in fulfilled_roles
        ]

        return bias_matrix, active_roles, unfulfilled

    # ------------------------------------------------------------------
    # Full symbolic analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        token_sequence: Sequence[str],
    ) -> DependencyAnalysis:
        tokens = list(token_sequence)
        bias, roles, missing = self.compute_paninian_attention_bias(tokens)

        verb_index = None
        verb = None
        for idx, token in enumerate(tokens):
            if token in self.verb_valency:
                verb_index = idx
                verb = token
                break

        assignments = [
            KarakaAssignment(
                role=role,
                token=data["token"],
                index=data["index"],
                root=data.get("root"),
            )
            for role, data in roles.items()
        ]

        bindings: List[DependencyBinding] = []

        if verb_index is not None:
            valency = self.verb_valency[verb]

            for assignment in assignments:
                if assignment.role in valency["demanded_roles"]:
                    score = 1.0
                elif assignment.role in valency["optional_roles"]:
                    score = 0.85
                else:
                    continue

                bindings.append(
                    DependencyBinding(
                        source_index=assignment.index,
                        target_index=verb_index,
                        relation=assignment.role,
                        score=score,
                        evidence=[
                            "explicit vibhakti-to-Kāraka mapping",
                            "verb Ākāṅkṣā / valency compatibility",
                        ],
                    )
                )

        return DependencyAnalysis(
            tokens=tokens,
            verb_index=verb_index,
            verb=verb,
            assignments=assignments,
            bindings=bindings,
            missing_required_roles=missing,
            attention_bias=bias.tolist(),
        )


# ============================================================================
# 3. Marathi bridge
# ============================================================================

class MarathiKarakaAdapter:
    """
    Conservative adapter from the File-5 Marathi parser to the File-8
    Kāraka representation.

    It uses only features actually exposed by the parser. If the parser has
    no explicit Kāraka/vibhakti evidence, the adapter does not manufacture a
    strong role label.
    """

    ROLE_MAP = {
        "instrumental": "Karaṇa",
        "dative": "Sampradāna",
        "ablative": "Apādāna",
        "genitive": "Sambandha",
        "locative": "Adhikaraṇa",
        "accusative": "Karma",
        "nominative": "Kartā",
    }

    def __init__(self, parser: Optional[Any] = None):
        if parser is None:
            if MarathiParser is None:
                raise RuntimeError(
                    "marathi_parser.py is required for Marathi analysis."
                )
            parser = MarathiParser()
        self.parser = parser

    @staticmethod
    def _value(value: Any) -> str:
        if hasattr(value, "value"):
            return str(value.value)
        return str(value) if value is not None else ""

    def candidate_roles(self, parsed: ParsedSentence) -> List[KarakaAssignment]:
        result: List[KarakaAssignment] = []

        for token in parsed.tokens:
            if token.kind.value != "word":
                continue

            analyses = parsed.analyses.get(token.index, [])
            if not analyses:
                continue

            # Preserve the parser's highest-ranked analysis as the bridge
            # representation. Full ambiguity remains available in parser data.
            analysis = analyses[0]
            features = analysis.features

            explicit = getattr(features, "karaka_candidate", None)
            case = self._value(getattr(features, "case", None)).lower()

            role = None
            score = 0.0
            evidence = ""

            if explicit:
                role = str(explicit)
                score = 0.94
                evidence = "parser-provided Kāraka candidate"
            elif case in self.ROLE_MAP:
                role = self.ROLE_MAP[case]
                score = 0.65
                evidence = f"morphological case={case}"

            if role is None:
                continue

            result.append(
                KarakaAssignment(
                    role=role,
                    token=token.text,
                    index=token.index,
                    root=getattr(features, "lemma", None),
                    score=score,
                    evidence=evidence,
                )
            )

        return result

    def analyze(self, text: str) -> Dict[str, Any]:
        parsed = self.parser.parse(text)
        candidates = self.candidate_roles(parsed)

        verb_indices = []
        for token in parsed.tokens:
            analysis = parsed.best_analysis(token.index)
            if analysis is None:
                continue
            pos = getattr(analysis.features, "pos", None)
            if hasattr(pos, "value"):
                pos = pos.value
            if pos == "verb":
                verb_indices.append(token.index)

        # Conservative structural bias: only candidate-role → nearest verb.
        n = len(parsed.tokens)
        bias = np.zeros((n, n), dtype=np.float32)
        bindings: List[DependencyBinding] = []

        for candidate in candidates:
            if not verb_indices:
                continue

            target = min(
                verb_indices,
                key=lambda i: abs(i - candidate.index),
            )

            strength = 5.0 * candidate.score
            bias[candidate.index, target] = strength
            bias[target, candidate.index] = strength

            bindings.append(
                DependencyBinding(
                    source_index=candidate.index,
                    target_index=target,
                    relation=candidate.role,
                    score=candidate.score,
                    evidence=[candidate.evidence],
                )
            )

        np.fill_diagonal(bias, 0.0)

        return {
            "record_type": "panini_marathi_karaka_dependency",
            "text": text,
            "tokens": [token.text for token in parsed.tokens],
            "verb_indices": verb_indices,
            "assignments": [a.to_dict() for a in candidates],
            "bindings": [b.to_dict() for b in bindings],
            "attention_bias": bias.tolist(),
            "metadata": {
                "language": "mr",
                "script": "Devanagari",
                "analysis_type": "conservative_parser_bridge",
                "ambiguity_preserved_in_source_parser": True,
            },
        }


# ============================================================================
# 4. Utility functions
# ============================================================================

def print_analysis(analysis: DependencyAnalysis) -> None:
    print("=" * 70)
    print("KĀRAKA SYNTACTIC DEPENDENCY INJECTION MATRIX")
    print("=" * 70)
    print(f"Tokens: {analysis.tokens}")
    print(f"Verb: {analysis.verb!r} @ index {analysis.verb_index}")

    print("\nResolved Kāraka slots:")
    if not analysis.assignments:
        print("  None")
    else:
        for item in analysis.assignments:
            print(
                f"  - {item.role:<12} -> '{item.token}' "
                f"(root={item.root}, score={item.score:.2f})"
            )

    print(
        "\nMissing required valencies: "
        f"{analysis.missing_required_roles}"
    )

    print("\nDependency bindings:")
    if not analysis.bindings:
        print("  None")
    else:
        for edge in analysis.bindings:
            print(
                f"  {analysis.tokens[edge.source_index]} "
                f"--{edge.relation}--> "
                f"{analysis.tokens[edge.target_index]} "
                f"score={edge.score:.2f}"
            )

    print("\nAttention bias matrix:")
    print(np.asarray(analysis.attention_bias, dtype=np.float32))


def matrix_checksum(matrix: Sequence[Sequence[float]]) -> str:
    arr = np.asarray(matrix, dtype=np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


# ============================================================================
# 5. Original structural demonstration
# ============================================================================

def run_dependency_simulation() -> DependencyAnalysis:
    engine = KarakaDependencyMatrix()

    # Deliberately scrambled physical sequence. Structural roles are obtained
    # from the inflection registry, not from word position.
    scrambled_sentence = [
        "hastena",
        "grāmam",
        "rāmāḥ",
        "gacchati",
    ]

    analysis = engine.analyze(scrambled_sentence)
    print_analysis(analysis)

    print("\nArchitectural interpretation:")
    print(
        "  Structural Kāraka relations are injected as a prior into "
        "the attention score space."
    )
    print(
        "  The +5.0 value is a prototype hyperparameter and should be "
        "benchmarked, learned, and ablated rather than assumed optimal."
    )
    print(
        f"  Matrix checksum: {matrix_checksum(analysis.attention_bias)}"
    )

    return analysis


# ============================================================================
# 6. CLI
# ============================================================================

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="karaka_dependency",
        description="Paninian Kāraka / dependency and attention-bias engine.",
    )

    parser.add_argument(
        "text",
        nargs="*",
        help="Sanskrit-style prototype tokens or Marathi sentence.",
    )
    parser.add_argument(
        "--marathi",
        action="store_true",
        help="Interpret text through marathi_parser.py.",
    )
    parser.add_argument(
        "--boost",
        type=float,
        default=5.0,
        help="Structural attention bias boost.",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Write the analysis as JSON.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the original scrambled Sanskrit demonstration.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run internal tests.",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_cli().parse_args(argv)

    if args.self_test:
        self_test()
        print("SELF_TEST_PASS")
        return 0

    if args.demo or not args.text:
        run_dependency_simulation()
        return 0

    if args.marathi:
        text = " ".join(args.text)
        payload = MarathiKarakaAdapter().analyze(text)

        print(json.dumps(payload, ensure_ascii=False, indent=2))

        if args.json:
            with open(args.json, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            print(f"JSON written to: {args.json}")
        return 0

    engine = KarakaDependencyMatrix(
        attention_boost=args.boost,
    )
    analysis = engine.analyze(args.text)
    print_analysis(analysis)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(analysis.to_dict(), handle, ensure_ascii=False, indent=2)
        print(f"JSON written to: {args.json}")

    return 0


# ============================================================================
# 7. Self-test
# ============================================================================

def self_test() -> None:
    engine = KarakaDependencyMatrix()

    tokens = ["hastena", "grāmam", "rāmāḥ", "gacchati"]
    analysis = engine.analyze(tokens)

    assert analysis.verb == "gacchati"
    assert analysis.verb_index == 3

    roles = {a.role: a for a in analysis.assignments}
    assert "Karaṇa" in roles
    assert "Karma" in roles
    assert "Kartā" in roles

    assert analysis.missing_required_roles == []

    # All three valid structural links should connect to the verb.
    assert len(analysis.bindings) == 3
    assert all(edge.target_index == 3 for edge in analysis.bindings)

    matrix = np.asarray(
        analysis.attention_bias,
        dtype=np.float32,
    )

    assert matrix.shape == (4, 4)
    assert matrix[0, 3] == 5.0
    assert matrix[3, 0] == 5.0
    assert matrix[1, 3] == 5.0
    assert matrix[3, 1] == 5.0
    assert matrix[2, 3] == 5.0
    assert matrix[3, 2] == 5.0
    assert np.all(np.diag(matrix) == 0.0)

    # Scrambling the tokens must not change which inflected item receives
    # which role; only the matrix coordinates move.
    scrambled = [
        "rāmāḥ",
        "gacchati",
        "hastena",
        "grāmam",
    ]
    scrambled_analysis = engine.analyze(scrambled)
    scrambled_roles = {
        a.role: a.token
        for a in scrambled_analysis.assignments
    }
    assert scrambled_roles["Kartā"] == "rāmāḥ"
    assert scrambled_roles["Karma"] == "grāmam"
    assert scrambled_roles["Karaṇa"] == "hastena"

    # No verb => flat matrix and no required-role assumptions.
    flat = engine.analyze(["rāmāḥ", "grāmam"])
    assert flat.verb_index is None
    assert flat.missing_required_roles == []
    assert np.all(
        np.asarray(flat.attention_bias) == 0.0
    )

    # Missing required role.
    missing = engine.analyze(["rāmāḥ", "gacchati"])
    assert "Karma" in missing.missing_required_roles

    # Boost is configurable.
    custom = KarakaDependencyMatrix(attention_boost=3.5)
    custom_analysis = custom.analyze(tokens)
    assert custom_analysis.attention_bias[0][3] == 3.5

    # JSON round-trip.
    payload = analysis.to_dict()
    encoded = json.dumps(payload, ensure_ascii=False)
    decoded = json.loads(encoded)
    assert decoded["record_type"] == "panini_karaka_dependency"
    assert decoded["attention_bias"][0][3] == 5.0

    # Marathi bridge should work when File 5 is present.
    if MarathiParser is not None:
        adapter = MarathiKarakaAdapter()
        marathi = adapter.analyze("राम पुस्तक घेऊन गेला.")
        assert marathi["record_type"] == "panini_marathi_karaka_dependency"
        assert len(marathi["tokens"]) == 5
        assert isinstance(marathi["attention_bias"], list)


if __name__ == "__main__":
    main()
