"""
panini_core.py
==============

Panini Language Machine — Core Intermediate Representation (PIR)

Purpose
-------
This module is the common foundation for the Paninian programs in the project.
It provides shared, serializable data structures for:

    Dhatu / Dhatu-Patha
    Pratyaya
    Anubandha markers
    Sutra rules
    Pratyahara sets
    Vibhakti
    Karaka roles
    Morphological tokens
    Derivation state and rule traces
    Semantic dependency graphs
    Neural constraint state

Design principle
----------------
The uploaded Paninian prototypes use the following recurring concepts:

    Dhatu + Pratyaya
        -> ordered grammatical processing
        -> Guna / morphological transformations
        -> Sandhi
        -> surface form

and:

    neural logits
        -> Paninian structural validation
        -> invalid paths masked
        -> valid output

This file deliberately contains DATA MODELS and LIGHTWEIGHT STATE OPERATIONS.
Grammar execution belongs in panini_engine.py and
ashtadhyayi_compiler.py.

No external packages are required.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json


# ============================================================================
# 1. Enumerations
# ============================================================================

class FeatureType(str, Enum):
    """Broad feature categories used by the symbolic representation."""

    MORPHOLOGY = "morphology"
    SYNTAX = "syntax"
    SEMANTICS = "semantics"
    PHONOLOGY = "phonology"
    CONSTRAINT = "constraint"


class TokenType(str, Enum):
    """Types of units that can appear in the factorized representation."""

    ROOT = "root"
    PRATYAYA = "pratyaya"
    WORD = "word"
    PUNCTUATION = "punctuation"
    UNKNOWN = "unknown"


class RuleKind(str, Enum):
    """Kinds of executable rules supported by the IR."""

    SUTRA = "sutra"
    ANUBANDHA = "anubandha"
    SANDHI = "sandhi"
    GUNA = "guna"
    VRDDHI = "vrddhi"
    VIBHAKTI = "vibhakti"
    KARAKA = "karaka"
    CONSTRAINT = "constraint"
    CUSTOM = "custom"


class KarakaRole(str, Enum):
    """
    Semantic roles used by the Kāraka layer.

    The source prototypes explicitly use roles such as Kartā, Karma and
    Karaṇa; the remaining common roles are represented so that the IR can
    support a fuller dependency graph without requiring another data model.
    """

    KARTA = "Kartā"
    KARMA = "Karma"
    KARANA = "Karaṇa"
    SAMPRADANA = "Sampradāna"
    APADANA = "Apādāna"
    ADHIKARANA = "Adhikaraṇa"
    HETU = "Hetu"
    UNKNOWN = "Unknown"


# ============================================================================
# 2. Lexical / grammatical records
# ============================================================================

@dataclass(frozen=True)
class Anubandha:
    """
    Metalinguistic marker associated with a Dhātu or Pratyaya.

    Markers are metadata. They are not necessarily retained in the surface
    form. The uploaded prototype uses markers to trigger or block
    transformations.
    """

    symbol: str
    description: str = ""
    triggers: Tuple[str, ...] = ()
    blocks: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Dhatu:
    """
    Dhātu-Pāṭha entry.

    Example fields represented in the uploaded prototype:
        root, meaning, markers, gana
    """

    root: str
    meaning: str = ""
    gana: str = ""
    markers: Tuple[str, ...] = ()
    anubandhas: Tuple[Anubandha, ...] = ()
    features: Dict[str, Any] = field(default_factory=dict)

    def marker_symbols(self) -> Tuple[str, ...]:
        """Return the compact marker symbols associated with this Dhātu."""
        if self.markers:
            return tuple(self.markers)
        return tuple(marker.symbol for marker in self.anubandhas)

    def has_marker(self, symbol: str) -> bool:
        return symbol in self.marker_symbols()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["anubandhas"] = [x.to_dict() for x in self.anubandhas]
        return data


@dataclass
class Pratyaya:
    """
    Pratyaya entry.

    The uploaded prototype models suffixes using properties such as:
        type, person, number, markers
    """

    form: str
    type: str = ""
    person: str = ""
    number: str = ""
    markers: Tuple[str, ...] = ()
    anubandhas: Tuple[Anubandha, ...] = ()
    features: Dict[str, Any] = field(default_factory=dict)

    def marker_symbols(self) -> Tuple[str, ...]:
        if self.markers:
            return tuple(self.markers)
        return tuple(marker.symbol for marker in self.anubandhas)

    def has_marker(self, symbol: str) -> bool:
        return symbol in self.marker_symbols()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["anubandhas"] = [x.to_dict() for x in self.anubandhas]
        return data


@dataclass(frozen=True)
class Pratyahara:
    """
    Symbolic sound-class / phonological set.

    Example codes from the project architecture include:
        AC, IK, HAL, YAN

    The actual membership is populated by the compiler or a grammar data
    loader; this class only defines the shared representation.
    """

    code: str
    members: Tuple[str, ...] = ()
    description: str = ""

    def contains(self, symbol: str) -> bool:
        return symbol in self.members

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Vibhakti:
    """Case / inflectional relation record."""

    name: str
    marker: str
    number: str = ""
    person: str = ""
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# 3. Grammar rule representation
# ============================================================================

@dataclass
class Sutra:
    """
    Executable-rule metadata for an Aṣṭādhyāyī Sūtra.

    `condition` and `transformation` are stored as symbolic descriptions or
    identifiers. Actual execution is deliberately delegated to the compiler
    and engine layers.
    """

    sutra_id: str
    text: str = ""
    kind: RuleKind = RuleKind.SUTRA
    adhikara: Optional[str] = None
    anuvrtti: Tuple[str, ...] = ()
    condition: Optional[str] = None
    transformation: Optional[str] = None
    priority: int = 0
    tags: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def applies_before(self, other: "Sutra") -> bool:
        """
        Simple priority comparison.

        A higher priority value wins. The full Vipratiṣedha / conflict
        resolution policy belongs to the compiler.
        """
        return self.priority > other.priority

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data


@dataclass
class RuleApplication:
    """One auditable application of a symbolic rule."""

    rule_id: str
    rule_kind: RuleKind
    input_form: str
    output_form: str
    reason: str = ""
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["rule_kind"] = self.rule_kind.value
        return data


# ============================================================================
# 4. Morphological / factorized token representation
# ============================================================================

@dataclass
class PaniniToken:
    """
    Factorized linguistic token.

    Instead of forcing an LLM tokenizer to memorize every inflected surface
    form, a word can be represented through its underlying root and suffix.
    """

    surface: str
    token_type: TokenType = TokenType.UNKNOWN
    root: Optional[str] = None
    suffix: Optional[str] = None
    normalized: Optional[str] = None
    valid: bool = False
    features: Dict[str, Any] = field(default_factory=dict)

    def factorized(self) -> Tuple[Optional[str], Optional[str]]:
        return self.root, self.suffix

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["token_type"] = self.token_type.value
        return data


@dataclass
class MorphologicalAnalysis:
    """Result of analyzing one surface word."""

    surface: str
    root: Optional[str] = None
    suffix: Optional[str] = None
    stem: Optional[str] = None
    vibhakti: Optional[Vibhakti] = None
    features: Dict[str, Any] = field(default_factory=dict)
    valid: bool = False
    confidence: float = 0.0
    rules: List[RuleApplication] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["vibhakti"] = (
            self.vibhakti.to_dict() if self.vibhakti is not None else None
        )
        data["rules"] = [rule.to_dict() for rule in self.rules]
        return data


# ============================================================================
# 5. Derivation state
# ============================================================================

@dataclass
class DerivationState:
    """
    Complete symbolic state during word derivation.

    This is the central object passed between the engine and compiler.
    """

    input_root: Optional[str] = None
    selected_pratyaya: Optional[str] = None
    intermediate_form: str = ""
    surface_form: Optional[str] = None

    dhatu: Optional[Dhatu] = None
    pratyaya: Optional[Pratyaya] = None

    features: Dict[str, Any] = field(default_factory=dict)
    active_rules: List[str] = field(default_factory=list)
    applied_rules: List[RuleApplication] = field(default_factory=list)
    blocked_rules: List[str] = field(default_factory=list)

    valid: bool = True
    errors: List[str] = field(default_factory=list)

    def add_rule(
        self,
        rule_id: str,
        rule_kind: RuleKind,
        input_form: str,
        output_form: str,
        reason: str = "",
        priority: int = 0,
        **metadata: Any,
    ) -> None:
        self.applied_rules.append(
            RuleApplication(
                rule_id=rule_id,
                rule_kind=rule_kind,
                input_form=input_form,
                output_form=output_form,
                reason=reason,
                priority=priority,
                metadata=metadata,
            )
        )

        if rule_id not in self.active_rules:
            self.active_rules.append(rule_id)

    def invalidate(self, reason: str) -> None:
        self.valid = False
        self.errors.append(reason)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_root": self.input_root,
            "selected_pratyaya": self.selected_pratyaya,
            "intermediate_form": self.intermediate_form,
            "surface_form": self.surface_form,
            "dhatu": self.dhatu.to_dict() if self.dhatu else None,
            "pratyaya": self.pratyaya.to_dict() if self.pratyaya else None,
            "features": self.features,
            "active_rules": list(self.active_rules),
            "applied_rules": [r.to_dict() for r in self.applied_rules],
            "blocked_rules": list(self.blocked_rules),
            "valid": self.valid,
            "errors": list(self.errors),
        }


# ============================================================================
# 6. Kāraka / semantic graph representation
# ============================================================================

@dataclass
class SemanticNode:
    """Node in the Kāraka dependency graph."""

    node_id: str
    surface: str
    role: KarakaRole = KarakaRole.UNKNOWN
    root: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["role"] = self.role.value
        return data


@dataclass
class SemanticEdge:
    """Directed relation between two semantic nodes."""

    source: str
    target: str
    relation: str
    weight: float = 1.0
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KarakaGraph:
    """
    Semantic dependency graph.

    The graph layer is kept independent of neural attention. A later module
    can convert this graph into an attention-bias matrix.
    """

    nodes: Dict[str, SemanticNode] = field(default_factory=dict)
    edges: List[SemanticEdge] = field(default_factory=list)
    root_verb: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: SemanticNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: SemanticEdge) -> None:
        self.edges.append(edge)

    def add_relation(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float = 1.0,
        **features: Any,
    ) -> None:
        self.add_edge(
            SemanticEdge(
                source=source,
                target=target,
                relation=relation,
                weight=weight,
                features=features,
            )
        )

    def neighbors(self, node_id: str) -> List[SemanticNode]:
        targets = {
            edge.target
            for edge in self.edges
            if edge.source == node_id
        }
        return [
            self.nodes[target]
            for target in targets
            if target in self.nodes
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {
                key: node.to_dict()
                for key, node in self.nodes.items()
            },
            "edges": [edge.to_dict() for edge in self.edges],
            "root_verb": self.root_verb,
            "features": self.features,
        }


# ============================================================================
# 7. Neural constraint representation
# ============================================================================

@dataclass
class ConstraintState:
    """
    Bridge between symbolic grammar and a neural next-token distribution.

    `valid_tokens` are approved by the Paninian rule layer.
    `masked_tokens` are structurally rejected.
    """

    context: List[str] = field(default_factory=list)
    valid_tokens: List[str] = field(default_factory=list)
    masked_tokens: List[str] = field(default_factory=list)
    logits: Dict[str, float] = field(default_factory=dict)
    constrained_logits: Dict[str, float] = field(default_factory=dict)

    def mask_invalid(self, invalid_value: float = float("-inf")) -> None:
        """Apply symbolic masking to the stored logits."""
        valid = set(self.valid_tokens)

        self.masked_tokens = []
        self.constrained_logits = {}

        for token, logit in self.logits.items():
            if token in valid:
                self.constrained_logits[token] = logit
            else:
                self.constrained_logits[token] = invalid_value
                self.masked_tokens.append(token)

    def best_token(self) -> Optional[str]:
        """Return the highest-scoring non-masked token."""
        if not self.constrained_logits:
            return None
        return max(self.constrained_logits, key=self.constrained_logits.get)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context": list(self.context),
            "valid_tokens": list(self.valid_tokens),
            "masked_tokens": list(self.masked_tokens),
            "logits": dict(self.logits),
            "constrained_logits": dict(self.constrained_logits),
        }


# ============================================================================
# 8. Corpus / structured example representation
# ============================================================================

@dataclass
class PaniniExample:
    """
    Training/export unit.

    This supports the structured JSONL representation produced later by
    panini_exporter.py.
    """

    text: str
    tokens: List[PaniniToken] = field(default_factory=list)
    morphology: List[MorphologicalAnalysis] = field(default_factory=list)
    karaka_graph: Optional[KarakaGraph] = None
    derivations: List[DerivationState] = field(default_factory=list)
    valid_syntax: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "tokens": [token.to_dict() for token in self.tokens],
            "morphology": [item.to_dict() for item in self.morphology],
            "karaka_graph": (
                self.karaka_graph.to_dict()
                if self.karaka_graph is not None
                else None
            ),
            "derivations": [item.to_dict() for item in self.derivations],
            "valid_syntax": self.valid_syntax,
            "metadata": self.metadata,
        }


# ============================================================================
# 9. Registry — common source of truth
# ============================================================================

@dataclass
class PaniniRegistry:
    """
    Canonical grammar registry shared by all modules.

    This prevents each prototype from independently maintaining its own
    vocabulary/root/suffix/rule dictionaries.
    """

    dhatus: Dict[str, Dhatu] = field(default_factory=dict)
    pratyayas: Dict[str, Pratyaya] = field(default_factory=dict)
    sutras: Dict[str, Sutra] = field(default_factory=dict)
    pratyaharas: Dict[str, Pratyahara] = field(default_factory=dict)
    vibhaktis: Dict[str, Vibhakti] = field(default_factory=dict)
    anubandhas: Dict[str, Anubandha] = field(default_factory=dict)

    def add_dhatu(self, dhatu: Dhatu) -> None:
        self.dhatus[dhatu.root] = dhatu

    def add_pratyaya(self, pratyaya: Pratyaya) -> None:
        self.pratyayas[pratyaya.form] = pratyaya

    def add_sutra(self, sutra: Sutra) -> None:
        self.sutras[sutra.sutra_id] = sutra

    def add_pratyahara(self, pratyahara: Pratyahara) -> None:
        self.pratyaharas[pratyahara.code] = pratyahara

    def add_vibhakti(self, vibhakti: Vibhakti) -> None:
        self.vibhaktis[vibhakti.name] = vibhakti

    def add_anubandha(self, anubandha: Anubandha) -> None:
        self.anubandhas[anubandha.symbol] = anubandha

    def get_dhatu(self, root: str) -> Dhatu:
        if root not in self.dhatus:
            raise KeyError(f"Unknown Dhatu: {root}")
        return self.dhatus[root]

    def get_pratyaya(self, form: str) -> Pratyaya:
        if form not in self.pratyayas:
            raise KeyError(f"Unknown Pratyaya: {form}")
        return self.pratyayas[form]

    def get_sutra(self, sutra_id: str) -> Sutra:
        if sutra_id not in self.sutras:
            raise KeyError(f"Unknown Sutra: {sutra_id}")
        return self.sutras[sutra_id]

    def summary(self) -> Dict[str, int]:
        return {
            "dhatus": len(self.dhatus),
            "pratyayas": len(self.pratyayas),
            "sutras": len(self.sutras),
            "pratyaharas": len(self.pratyaharas),
            "vibhaktis": len(self.vibhaktis),
            "anubandhas": len(self.anubandhas),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dhatus": {
                key: value.to_dict()
                for key, value in self.dhatus.items()
            },
            "pratyayas": {
                key: value.to_dict()
                for key, value in self.pratyayas.items()
            },
            "sutras": {
                key: value.to_dict()
                for key, value in self.sutras.items()
            },
            "pratyaharas": {
                key: value.to_dict()
                for key, value in self.pratyaharas.items()
            },
            "vibhaktis": {
                key: value.to_dict()
                for key, value in self.vibhaktis.items()
            },
            "anubandhas": {
                key: value.to_dict()
                for key, value in self.anubandhas.items()
            },
        }


# ============================================================================
# 10. Serialization helpers
# ============================================================================

def to_json(obj: Any, *, indent: int = 2) -> str:
    """Serialize a Panini IR object or ordinary mapping to JSON."""

    if hasattr(obj, "to_dict"):
        payload = obj.to_dict()
    elif hasattr(obj, "__dataclass_fields__"):
        payload = asdict(obj)
    else:
        payload = obj

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=indent,
        default=str,
    )


def save_json(obj: Any, path: str) -> None:
    """Save any serializable Panini IR object as UTF-8 JSON."""

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(to_json(obj))
        handle.write("\n")


# ============================================================================
# 11. Prototype registry
# ============================================================================

def build_prototype_registry() -> PaniniRegistry:
    """
    Build the small registry used by the current prototypes.

    The uploaded code explicitly uses roots such as bhū, paṭh, gam and nī,
    and suffixes such as ti, anti and ta. This registry centralizes those
    entries so later files can import them instead of redefining them.
    """

    registry = PaniniRegistry()

    # Anubandha markers appearing in the prototype family.
    registry.add_anubandha(
        Anubandha(
            symbol="i",
            description="Prototype marker associated with vowel/Guna behavior",
            triggers=("guna",),
        )
    )
    registry.add_anubandha(
        Anubandha(
            symbol="t",
            description="Prototype marker associated with suffix behavior",
        )
    )
    registry.add_anubandha(
        Anubandha(
            symbol="k",
            description="Prototype Kit marker; may block Guna",
            blocks=("guna",),
        )
    )

    # Dhatu-Patha prototype entries.
    registry.add_dhatu(
        Dhatu(
            root="bhū",
            meaning="to be",
            gana="Bhvadi",
            markers=(),
        )
    )
    registry.add_dhatu(
        Dhatu(
            root="paṭh",
            meaning="to read",
            gana="Bhvadi",
            markers=("i",),
        )
    )
    registry.add_dhatu(
        Dhatu(
            root="gam",
            meaning="to go",
            gana="Bhvadi",
            markers=("t",),
        )
    )
    registry.add_dhatu(
        Dhatu(
            root="nī",
            meaning="to lead",
            gana="Bhvadi",
            markers=("i",),
        )
    )

    # Prototype Pratyayas.
    registry.add_pratyaya(
        Pratyaya(
            form="ti",
            type="present",
            person="3rd",
            number="singular",
        )
    )
    registry.add_pratyaya(
        Pratyaya(
            form="anti",
            type="present",
            person="3rd",
            number="plural",
        )
    )
    registry.add_pratyaya(
        Pratyaya(
            form="ta",
            type="past_participle",
            markers=("k",),
        )
    )

    # Prototype phonological sets. Membership is intentionally compact;
    # the scalable compiler will own the complete inventory.
    registry.add_pratyahara(
        Pratyahara(
            code="AC",
            members=("a", "ā", "i", "ī", "u", "ū", "ṛ", "e", "o"),
            description="Prototype vowel class",
        )
    )
    registry.add_pratyahara(
        Pratyahara(
            code="IK",
            members=("i", "ī", "u", "ū", "ṛ"),
            description="Prototype i/u/ṛ vowel class",
        )
    )
    registry.add_pratyahara(
        Pratyahara(
            code="HAL",
            members=(
                "k", "kh", "g", "gh", "ṅ",
                "c", "ch", "j", "jh", "ñ",
                "ṭ", "ṭh", "ḍ", "ḍh", "ṇ",
                "t", "th", "d", "dh", "n",
                "p", "ph", "b", "bh", "m",
                "y", "r", "l", "v", "ś", "ṣ", "s", "h",
            ),
            description="Prototype consonant class",
        )
    )
    registry.add_pratyahara(
        Pratyahara(
            code="YAN",
            members=("y", "v", "r", "l"),
            description="Prototype semivowel class",
        )
    )

    return registry


# ============================================================================
# 12. Self-test
# ============================================================================

def self_test() -> None:
    """Run lightweight integrity checks without external dependencies."""

    registry = build_prototype_registry()

    assert "bhū" in registry.dhatus
    assert "gam" in registry.dhatus
    assert "nī" in registry.dhatus

    assert "ti" in registry.pratyayas
    assert "anti" in registry.pratyayas
    assert "ta" in registry.pratyayas

    assert registry.dhatus["gam"].has_marker("t")
    assert registry.pratyayas["ta"].has_marker("k")

    assert registry.pratyaharas["IK"].contains("i")
    assert registry.pratyaharas["HAL"].contains("m")
    assert not registry.pratyaharas["IK"].contains("a")

    state = DerivationState(
        input_root="bhū",
        selected_pratyaya="ti",
        intermediate_form="bho",
    )

    state.add_rule(
        rule_id="prototype-guna",
        rule_kind=RuleKind.GUNA,
        input_form="bhū",
        output_form="bho",
        reason="Prototype Guna transformation",
    )

    assert state.valid
    assert len(state.applied_rules) == 1

    constraint = ConstraintState(
        context=["bhū"],
        valid_tokens=["ti"],
        logits={"ti": 2.5, "anti": 3.5, "gam": 1.2},
    )
    constraint.mask_invalid()

    assert constraint.constrained_logits["ti"] == 2.5
    assert constraint.constrained_logits["anti"] == float("-inf")
    assert constraint.constrained_logits["gam"] == float("-inf")
    assert constraint.best_token() == "ti"


# ============================================================================
# 13. Demonstration
# ============================================================================

def demo() -> None:
    registry = build_prototype_registry()

    print("=" * 72)
    print("PANINI LANGUAGE MACHINE — PANINIAN INTERMEDIATE REPRESENTATION")
    print("=" * 72)

    print("\nRegistry summary:")
    for key, value in registry.summary().items():
        print(f"  {key:<14}: {value}")

    print("\nDhatu-Patha:")
    for root, dhatu in registry.dhatus.items():
        print(
            f"  {root:<5} | meaning={dhatu.meaning:<12} "
            f"| gana={dhatu.gana:<8} | markers={dhatu.marker_symbols()}"
        )

    print("\nPratyayas:")
    for form, pratyaya in registry.pratyayas.items():
        print(
            f"  {form:<5} | type={pratyaya.type:<16} "
            f"| person={pratyaya.person:<5} "
            f"| number={pratyaya.number:<9} "
            f"| markers={pratyaya.marker_symbols()}"
        )

    state = DerivationState(
        input_root="bhū",
        selected_pratyaya="ti",
        intermediate_form="bho",
    )
    state.add_rule(
        rule_id="7.3.prototype.guna",
        rule_kind=RuleKind.GUNA,
        input_form="bhū",
        output_form="bho",
        reason="Prototype vowel strengthening step",
    )

    print("\nExample DerivationState:")
    print(to_json(state))

    constraint = ConstraintState(
        context=["bhū"],
        valid_tokens=["ti"],
        logits={
            "anti": 3.50,
            "ti": 2.50,
            "gam": 1.20,
        },
    )
    constraint.mask_invalid()

    print("\nExample ConstraintState:")
    print(to_json(constraint))

    print("\nSelf-test:")
    self_test()
    print("  PASS")


if __name__ == "__main__":
    demo()
