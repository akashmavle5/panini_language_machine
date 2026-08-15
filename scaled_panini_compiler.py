"""
scaled_panini_compiler.py
=========================

Panini Language Machine — Scaled Aṣṭādhyāyī Compiler

File 4/14.

Purpose
-------
Scale the declarative compiler in `ashtadhyayi_compiler.py` from a small
prototype rule pack to a grammar-engineering layer suitable for thousands of
Sūtra specifications.

This file is responsible for:

1. loading rule records from JSON / JSONL / Python mappings
2. validating large rule corpora
3. assigning deterministic source order
4. expanding Adhikāra / Anuvṛtti inheritance
5. building multiple lookup indexes
6. resolving rule conflicts deterministically
7. compiling executable rule bundles
8. producing compact grammar metadata for downstream symbolic/neural use

Source fidelity
---------------
The preceding prototype files establish the concepts of:
    Sūtra
    Adhikāra
    Anuvṛtti
    Pratyāhāra
    Anubandha
    priority
    deterministic derivation
    executable transformations

This file does not claim that an absent machine-readable Aṣṭādhyāyī corpus
has been reconstructed. It provides the scaling machinery needed when that
corpus is supplied.

Design principle
----------------
Do not make the runtime repeatedly scan all rules.

Instead:

    large rule corpus
          ↓
    normalize
          ↓
    validate
          ↓
    inherit context
          ↓
    compile
          ↓
    indexes
          ↓
    candidate retrieval
          ↓
    conflict resolution
          ↓
    executable bundle

The resulting architecture is intended to support later:
    neuro_symbolic_panini.py
    paninian_english_llm.py
    neuro_symbolic_trainer.py
    paninian_vs_llm_benchmarker.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
import hashlib
import json
import re

from panini_core import PaniniRegistry, RuleKind, build_prototype_registry
from ashtadhyayi_compiler import (
    Action,
    AshtadhyayiCompiler,
    CompiledRule,
    Condition,
    Operator,
    RuleIndex,
    RuleSpec,
    Transformation,
)


# ============================================================================
# 1. Scaled compiler configuration
# ============================================================================

@dataclass(frozen=True)
class CompilerConfig:
    """
    Runtime/compiler policy.

    `priority_weight` is deliberately explicit: it makes ranking policy
    inspectable rather than hiding it inside an arbitrary sort.
    """

    priority_weight: int = 1_000_000
    source_order_weight: int = 1_000
    specificity_weight: int = 10
    max_candidates: int = 256
    strict_validation: bool = True
    deduplicate_rules: bool = True
    preserve_source_order: bool = True


# ============================================================================
# 2. Diagnostics
# ============================================================================

class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Diagnostic:
    severity: Severity
    code: str
    message: str
    sutra_id: Optional[str] = None
    source_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "sutra_id": self.sutra_id,
            "source_index": self.source_index,
        }


@dataclass
class CompilationReport:
    source_rules: int = 0
    normalized_rules: int = 0
    compiled_rules: int = 0
    duplicate_rules: int = 0
    warnings: int = 0
    errors: int = 0
    diagnostics: List[Diagnostic] = field(default_factory=list)

    def add(self, diagnostic: Diagnostic) -> None:
        self.diagnostics.append(diagnostic)

        if diagnostic.severity == Severity.WARNING:
            self.warnings += 1
        elif diagnostic.severity == Severity.ERROR:
            self.errors += 1

    @property
    def success(self) -> bool:
        return self.errors == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_rules": self.source_rules,
            "normalized_rules": self.normalized_rules,
            "compiled_rules": self.compiled_rules,
            "duplicate_rules": self.duplicate_rules,
            "warnings": self.warnings,
            "errors": self.errors,
            "success": self.success,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


# ============================================================================
# 3. Normalized corpus representation
# ============================================================================

@dataclass(frozen=True)
class RuleFingerprint:
    """
    Stable semantic fingerprint used for duplicate detection.

    It deliberately excludes source order and compiled IDs.
    """

    value: str

    @staticmethod
    def from_rule(rule: RuleSpec) -> "RuleFingerprint":
        payload = {
            "sutra_id": rule.sutra_id,
            "kind": rule.kind.value,
            "conditions": [
                condition.to_dict()
                for condition in rule.conditions
            ],
            "transformations": [
                transformation.to_dict()
                for transformation in rule.transformations
            ],
            "adhikara": rule.adhikara,
            "anuvrtti": list(rule.anuvrtti),
            "tags": list(rule.tags),
        }

        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

        return RuleFingerprint(
            hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        )


@dataclass
class NormalizedRule:
    spec: RuleSpec
    source_index: int
    fingerprint: RuleFingerprint
    effective_adhikara: Optional[str] = None
    effective_anuvrtti: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_index": self.source_index,
            "fingerprint": self.fingerprint.value,
            "effective_adhikara": self.effective_adhikara,
            "effective_anuvrtti": list(self.effective_anuvrtti),
            "spec": self.spec.to_dict(),
        }


# ============================================================================
# 4. Corpus loader
# ============================================================================

class RuleCorpusLoader:
    """
    Load large declarative corpora.

    Supported input:
        * iterable of RuleSpec
        * iterable of dictionaries
        * JSON array
        * JSON object containing `rules`
        * JSONL / NDJSON
    """

    @staticmethod
    def _mapping_to_rule(item: Mapping[str, Any]) -> RuleSpec:
        conditions = [
            Condition(
                field=condition["field"],
                operator=Operator(condition["operator"]),
                value=condition.get("value"),
            )
            for condition in item.get("conditions", [])
        ]

        transformations = [
            Transformation(
                action=Action(transformation["action"]),
                target=transformation.get("target", "stem"),
                value=transformation.get("value"),
                replacement=transformation.get("replacement"),
            )
            for transformation in item.get("transformations", [])
        ]

        return RuleSpec(
            sutra_id=str(item["sutra_id"]),
            text=item.get("text", ""),
            kind=RuleKind(item.get("kind", RuleKind.SUTRA.value)),
            conditions=conditions,
            transformations=transformations,
            adhikara=item.get("adhikara"),
            anuvrtti=tuple(item.get("anuvrtti", [])),
            priority=int(item.get("priority", 0)),
            tags=tuple(item.get("tags", [])),
            metadata=dict(item.get("metadata", {})),
        )

    @classmethod
    def normalize_records(
        cls,
        records: Iterable[Any],
    ) -> List[RuleSpec]:
        result: List[RuleSpec] = []

        for record in records:
            if isinstance(record, RuleSpec):
                result.append(record)
            elif isinstance(record, Mapping):
                result.append(cls._mapping_to_rule(record))
            else:
                raise TypeError(
                    "Rule corpus entries must be RuleSpec or mapping objects"
                )

        return result

    @classmethod
    def load_file(cls, path: str | Path) -> List[RuleSpec]:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        suffix = path.suffix.lower()

        if suffix in {".jsonl", ".ndjson"}:
            records = []
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSONL at {path}:{line_number}: {exc}"
                        ) from exc

            return cls.normalize_records(records)

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, Mapping):
            payload = payload.get("rules", payload)

        if not isinstance(payload, list):
            raise ValueError(
                "JSON rule corpus must be an array or an object containing "
                "`rules`"
            )

        return cls.normalize_records(payload)


# ============================================================================
# 5. Large-corpus indexes
# ============================================================================

class ScaledRuleIndex:
    """
    Index optimized for candidate retrieval.

    Index dimensions:
        * root
        * suffix
        * tag
        * kind
        * adhikara
        * feature fields used by conditions
        * unconditional rules

    The index is intentionally additive. If a rule cannot be indexed
    narrowly, it is retained in the fallback bucket rather than discarded.
    """

    def __init__(self) -> None:
        self.rules: Dict[str, NormalizedRule] = {}

        self.by_root: Dict[str, List[str]] = {}
        self.by_suffix: Dict[str, List[str]] = {}
        self.by_tag: Dict[str, List[str]] = {}
        self.by_kind: Dict[str, List[str]] = {}
        self.by_adhikara: Dict[str, List[str]] = {}

        self.by_field_value: Dict[
            Tuple[str, str],
            List[str],
        ] = {}

        self.fallback: List[str] = []

    @staticmethod
    def _append(
        mapping: Dict[Any, List[str]],
        key: Any,
        rule_id: str,
    ) -> None:
        bucket = mapping.setdefault(key, [])
        if rule_id not in bucket:
            bucket.append(rule_id)

    def add(self, rule: NormalizedRule) -> None:
        rule_id = rule.spec.sutra_id

        self.rules[rule_id] = rule

        indexed = False

        for condition in rule.spec.conditions:
            if (
                condition.operator == Operator.EQUALS
                and condition.field == "root"
            ):
                self._append(
                    self.by_root,
                    str(condition.value),
                    rule_id,
                )
                indexed = True

            elif (
                condition.operator == Operator.EQUALS
                and condition.field == "suffix"
            ):
                self._append(
                    self.by_suffix,
                    str(condition.value),
                    rule_id,
                )
                indexed = True

            elif (
                condition.operator == Operator.EQUALS
                and condition.field.startswith("feature:")
            ):
                self._append(
                    self.by_field_value,
                    (
                        condition.field,
                        json.dumps(
                            condition.value,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                    rule_id,
                )
                indexed = True

        for tag in rule.spec.tags:
            self._append(self.by_tag, tag, rule_id)
            indexed = True

        self._append(
            self.by_kind,
            rule.spec.kind.value,
            rule_id,
        )

        if rule.effective_adhikara:
            self._append(
                self.by_adhikara,
                rule.effective_adhikara,
                rule_id,
            )

        if not indexed:
            self.fallback.append(rule_id)

    def _rules(self, ids: Iterable[str]) -> List[NormalizedRule]:
        return [
            self.rules[rule_id]
            for rule_id in ids
            if rule_id in self.rules
        ]

    def candidates(
        self,
        *,
        root: Optional[str] = None,
        suffix: Optional[str] = None,
        tags: Sequence[str] = (),
        kind: Optional[str] = None,
        adhikara: Optional[str] = None,
        limit: int = 256,
    ) -> List[NormalizedRule]:
        buckets: List[set[str]] = []

        if root is not None and root in self.by_root:
            buckets.append(set(self.by_root[root]))

        if suffix is not None and suffix in self.by_suffix:
            buckets.append(set(self.by_suffix[suffix]))

        for tag in tags:
            if tag in self.by_tag:
                buckets.append(set(self.by_tag[tag]))

        if kind is not None and kind in self.by_kind:
            buckets.append(set(self.by_kind[kind]))

        if adhikara is not None and adhikara in self.by_adhikara:
            buckets.append(set(self.by_adhikara[adhikara]))

        if buckets:
            # Candidate union rather than intersection is intentional:
            # conditions not represented by an index must still be evaluated
            # by the final RuleSpec.matches() stage.
            ids: set[str] = set()
            for bucket in buckets:
                ids.update(bucket)

            ids.update(self.fallback)

        else:
            ids = set(self.rules)

        result = [self.rules[rule_id] for rule_id in ids]

        result.sort(
            key=lambda item: (
                -item.spec.priority,
                -len(item.spec.conditions),
                item.source_index,
            )
        )

        return result[:limit]

    def summary(self) -> Dict[str, int]:
        return {
            "rules": len(self.rules),
            "root_keys": len(self.by_root),
            "suffix_keys": len(self.by_suffix),
            "tag_keys": len(self.by_tag),
            "kind_keys": len(self.by_kind),
            "adhikara_keys": len(self.by_adhikara),
            "feature_keys": len(self.by_field_value),
            "fallback_rules": len(self.fallback),
        }


# ============================================================================
# 6. Conflict resolution
# ============================================================================

@dataclass
class RankedRule:
    rule: NormalizedRule
    score: int
    reasons: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sutra_id": self.rule.spec.sutra_id,
            "score": self.score,
            "reasons": list(self.reasons),
        }


class RuleConflictResolver:
    """
    Deterministic ranking policy.

    Ranking is not presented as a claim about every subtlety of Pāṇinian
    rule precedence. It is an explicit engineering policy that provides a
    stable runtime ordering until a richer source-level precedence model is
    supplied.
    """

    def __init__(self, config: CompilerConfig):
        self.config = config

    def rank(
        self,
        rules: Iterable[NormalizedRule],
        *,
        context: Mapping[str, Any],
    ) -> List[RankedRule]:
        ranked: List[RankedRule] = []

        for rule in rules:
            if not rule.spec.matches(context):
                continue

            score = (
                rule.spec.priority * self.config.priority_weight
                + len(rule.spec.conditions) * self.config.specificity_weight
                + (
                    self.config.source_order_weight
                    * max(0, 100000 - rule.source_index)
                )
            )

            reasons = (
                f"priority={rule.spec.priority}",
                f"specificity={len(rule.spec.conditions)}",
                f"source_index={rule.source_index}",
            )

            ranked.append(
                RankedRule(
                    rule=rule,
                    score=score,
                    reasons=reasons,
                )
            )

        ranked.sort(
            key=lambda item: (
                -item.score,
                item.rule.source_index,
                item.rule.spec.sutra_id,
            )
        )

        return ranked


# ============================================================================
# 7. Scaled compiler
# ============================================================================

class ScaledPaniniCompiler:
    """
    Large-corpus front-end around AshtadhyayiCompiler.

    It deliberately keeps the original compiler as the canonical executable
    rule representation and adds scaling facilities around it.
    """

    def __init__(
        self,
        registry: Optional[PaniniRegistry] = None,
        *,
        config: Optional[CompilerConfig] = None,
    ) -> None:
        self.registry = registry or build_prototype_registry()
        self.config = config or CompilerConfig()

        self.base_compiler = AshtadhyayiCompiler(self.registry)

        self.normalized_rules: List[NormalizedRule] = []
        self.index = ScaledRuleIndex()
        self.resolver = RuleConflictResolver(self.config)

        self.report = CompilationReport()
        self.compiled: bool = False

    # ------------------------------------------------------------------
    # Corpus ingestion
    # ------------------------------------------------------------------

    def add_rules(self, rules: Iterable[RuleSpec]) -> None:
        incoming = list(rules)

        self.report.source_rules += len(incoming)

        for rule in incoming:
            self.base_compiler.add_rule(rule)

        self.compiled = False

    def load_file(self, path: str | Path) -> None:
        self.add_rules(RuleCorpusLoader.load_file(path))

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize(self) -> List[NormalizedRule]:
        source = list(self.base_compiler.source_rules)
        normalized: List[NormalizedRule] = []

        active_adhikara: Optional[str] = None
        inherited_terms: List[str] = []

        fingerprints: set[str] = set()

        for source_index, rule in enumerate(source):
            if rule.adhikara is not None:
                active_adhikara = rule.adhikara
                inherited_terms = list(rule.anuvrtti)

            for term in rule.anuvrtti:
                if term not in inherited_terms:
                    inherited_terms.append(term)

            effective_anuvrtti = tuple(dict.fromkeys(inherited_terms))

            effective_adhikara = rule.adhikara or active_adhikara

            # Materialize inherited context into a new immutable RuleSpec.
            normalized_spec = RuleSpec(
                sutra_id=rule.sutra_id,
                text=rule.text,
                kind=rule.kind,
                conditions=list(rule.conditions),
                transformations=list(rule.transformations),
                adhikara=effective_adhikara,
                anuvrtti=effective_anuvrtti,
                priority=rule.priority,
                tags=rule.tags,
                metadata={
                    **rule.metadata,
                    "source_index": source_index,
                    "compiled_adhikara": effective_adhikara,
                },
            )

            fingerprint = RuleFingerprint.from_rule(normalized_spec)

            if (
                self.config.deduplicate_rules
                and fingerprint.value in fingerprints
            ):
                self.report.duplicate_rules += 1
                self.report.add(
                    Diagnostic(
                        severity=Severity.WARNING,
                        code="DUPLICATE_RULE",
                        message="Duplicate semantic rule skipped",
                        sutra_id=rule.sutra_id,
                        source_index=source_index,
                    )
                )
                continue

            fingerprints.add(fingerprint.value)

            normalized.append(
                NormalizedRule(
                    spec=normalized_spec,
                    source_index=source_index,
                    fingerprint=fingerprint,
                    effective_adhikara=effective_adhikara,
                    effective_anuvrtti=effective_anuvrtti,
                )
            )

        self.report.normalized_rules = len(normalized)

        return normalized

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_normalized(
        self,
        rules: Sequence[NormalizedRule],
    ) -> None:
        seen_ids: set[str] = set()

        for rule in rules:
            sutra_id = rule.spec.sutra_id

            if sutra_id in seen_ids:
                self.report.add(
                    Diagnostic(
                        severity=Severity.ERROR,
                        code="DUPLICATE_ID",
                        message="Duplicate Sūtra identifier",
                        sutra_id=sutra_id,
                        source_index=rule.source_index,
                    )
                )

            seen_ids.add(sutra_id)

            if not rule.spec.transformations:
                self.report.add(
                    Diagnostic(
                        severity=Severity.ERROR,
                        code="NO_TRANSFORMATION",
                        message="Rule has no executable transformation",
                        sutra_id=sutra_id,
                        source_index=rule.source_index,
                    )
                )

            if rule.spec.priority < 0:
                self.report.add(
                    Diagnostic(
                        severity=Severity.WARNING,
                        code="NEGATIVE_PRIORITY",
                        message="Negative priority is allowed but unusual",
                        sutra_id=sutra_id,
                        source_index=rule.source_index,
                    )
                )

            if not rule.spec.text:
                self.report.add(
                    Diagnostic(
                        severity=Severity.INFO,
                        code="NO_TEXT",
                        message="Rule has no source text",
                        sutra_id=sutra_id,
                        source_index=rule.source_index,
                    )
                )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def compile(self) -> CompilationReport:
        self.report = CompilationReport(
            source_rules=len(self.base_compiler.source_rules)
        )

        self.normalized_rules = self._normalize()
        self._validate_normalized(self.normalized_rules)

        if (
            self.config.strict_validation
            and self.report.errors
        ):
            raise ValueError(
                "Scaled Panini compilation failed:\n"
                + "\n".join(
                    f"[{d.code}] {d.message}"
                    for d in self.report.diagnostics
                    if d.severity == Severity.ERROR
                )
            )

        self.index = ScaledRuleIndex()

        for normalized in self.normalized_rules:
            self.index.add(normalized)

        # Keep AshtadhyayiCompiler as the executable registry-facing layer.
        self.base_compiler.source_rules = [
            normalized.spec
            for normalized in self.normalized_rules
        ]
        self.base_compiler.compile(strict=self.config.strict_validation)

        self.report.compiled_rules = len(self.normalized_rules)
        self.compiled = True

        return self.report

    # ------------------------------------------------------------------
    # Candidate retrieval
    # ------------------------------------------------------------------

    def candidate_rules(
        self,
        *,
        root: Optional[str] = None,
        suffix: Optional[str] = None,
        tags: Sequence[str] = (),
        kind: Optional[str] = None,
        adhikara: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[NormalizedRule]:
        if not self.compiled:
            self.compile()

        return self.index.candidates(
            root=root,
            suffix=suffix,
            tags=tags,
            kind=kind,
            adhikara=adhikara,
            limit=limit or self.config.max_candidates,
        )

    # ------------------------------------------------------------------
    # Rule resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        context: Mapping[str, Any],
        *,
        root: Optional[str] = None,
        suffix: Optional[str] = None,
        tags: Sequence[str] = (),
        kind: Optional[str] = None,
        adhikara: Optional[str] = None,
    ) -> List[RankedRule]:
        candidates = self.candidate_rules(
            root=root,
            suffix=suffix,
            tags=tags,
            kind=kind,
            adhikara=adhikara,
        )

        return self.resolver.rank(
            candidates,
            context=context,
        )

    def best_rule(
        self,
        context: Mapping[str, Any],
        **filters: Any,
    ) -> Optional[RankedRule]:
        ranked = self.resolve(context, **filters)
        return ranked[0] if ranked else None

    # ------------------------------------------------------------------
    # Runtime context
    # ------------------------------------------------------------------

    def build_context(
        self,
        *,
        root: Optional[str] = None,
        suffix: Optional[str] = None,
        stem: Optional[str] = None,
        surface: Optional[str] = None,
        features: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.base_compiler.build_context(
            root=root,
            suffix=suffix,
            stem=stem,
            surface=surface,
            features=features,
        )

    # ------------------------------------------------------------------
    # Derivation
    # ------------------------------------------------------------------

    def derive(
        self,
        root: str,
        suffix: str,
        *,
        apply_base_engine: bool = False,
    ):
        """
        Execute through the compiled Ashtadhyayi compiler.

        The default is False because the scaled compiler's responsibility is
        to demonstrate deterministic corpus selection. The later integration
        layer can explicitly combine this with the morphology engine.
        """
        if not self.compiled:
            self.compile()

        return self.base_compiler.derive(
            root,
            suffix,
            apply_base_engine=apply_base_engine,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_manifest(self) -> Dict[str, Any]:
        if not self.compiled:
            self.compile()

        return {
            "compiler": "scaled_panini_compiler",
            "version": "0.1",
            "compiled": self.compiled,
            "config": {
                key: getattr(self.config, key)
                for key in self.config.__dataclass_fields__
            },
            "report": self.report.to_dict(),
            "index": self.index.summary(),
            "rules": [
                rule.to_dict()
                for rule in self.normalized_rules
            ],
        }

    def export_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.export_manifest(),
            ensure_ascii=False,
            indent=indent,
            default=str,
        )


# ============================================================================
# 8. Prototype scaled corpus
# ============================================================================

def prototype_scaled_rules() -> List[RuleSpec]:
    """
    Return a larger structured prototype corpus.

    It deliberately reuses the rule concepts established by File 3 while
    introducing additional indexed conditions so the scaling machinery can
    be tested.
    """
    return [
        RuleSpec(
            sutra_id="scaled.guna.bhu",
            text="Prototype Guṇa rule for bhū",
            kind=RuleKind.GUNA,
            conditions=[
                Condition(
                    field="root",
                    operator=Operator.EQUALS,
                    value="bhū",
                ),
            ],
            transformations=[
                Transformation(
                    action=Action.REPLACE,
                    value="bhū",
                    replacement="bho",
                )
            ],
            adhikara="scaled-guna",
            anuvrtti=("guna",),
            priority=100,
            tags=("guna", "bhu"),
        ),
        RuleSpec(
            sutra_id="scaled.guna.ni",
            text="Prototype Guṇa rule for nī",
            kind=RuleKind.GUNA,
            conditions=[
                Condition(
                    field="root",
                    operator=Operator.EQUALS,
                    value="nī",
                ),
            ],
            transformations=[
                Transformation(
                    action=Action.REPLACE,
                    value="nī",
                    replacement="ne",
                )
            ],
            adhikara="scaled-guna",
            anuvrtti=("guna",),
            priority=100,
            tags=("guna", "ni"),
        ),
        RuleSpec(
            sutra_id="scaled.gam.present",
            text="Prototype present transformation for gam",
            kind=RuleKind.SUTRA,
            conditions=[
                Condition(
                    field="root",
                    operator=Operator.EQUALS,
                    value="gam",
                ),
                Condition(
                    field="pratyaya_type",
                    operator=Operator.EQUALS,
                    value="present",
                ),
            ],
            transformations=[
                Transformation(
                    action=Action.REPLACE,
                    value="gam",
                    replacement="gacch",
                )
            ],
            adhikara="scaled-present",
            anuvrtti=("present",),
            priority=110,
            tags=("present", "gam"),
        ),
        RuleSpec(
            sutra_id="scaled.constraint.k",
            text="Prototype Kit-marker constraint",
            kind=RuleKind.CONSTRAINT,
            conditions=[
                Condition(
                    field="markers",
                    operator=Operator.HAS_MARKER,
                    value="k",
                ),
            ],
            transformations=[
                Transformation(action=Action.IDENTITY)
            ],
            adhikara="scaled-guna",
            anuvrtti=("guna",),
            priority=1000,
            tags=("constraint", "marker"),
        ),
        RuleSpec(
            sutra_id="scaled.surface.vowel",
            text="Prototype surface-vowel rule",
            kind=RuleKind.SUTRA,
            conditions=[
                Condition(
                    field="surface",
                    operator=Operator.MATCHES,
                    value=r"^[aeiouāīūṛḷeo]",
                ),
            ],
            transformations=[
                Transformation(action=Action.IDENTITY)
            ],
            adhikara="scaled-surface",
            anuvrtti=("surface",),
            priority=10,
            tags=("surface", "vowel"),
        ),
    ]


def build_scaled_prototype_compiler() -> ScaledPaniniCompiler:
    compiler = ScaledPaniniCompiler()

    # We intentionally don't include File 3's prototype rules here because
    # this file tests the independent scaling layer.
    compiler.add_rules(prototype_scaled_rules())
    compiler.compile()

    return compiler


# ============================================================================
# 9. Corpus generator for performance testing
# ============================================================================

def replicate_rules(
    rules: Sequence[RuleSpec],
    count: int,
) -> List[RuleSpec]:
    """
    Generate deterministic synthetic corpus size for compiler benchmarks.

    This is a tooling function, not a claim about authentic Aṣṭādhyāyī
    content. IDs are suffixed so duplicate detection can be tested separately
    from semantic equivalence.
    """
    if count <= 0:
        return []

    result: List[RuleSpec] = []

    for index in range(count):
        template = rules[index % len(rules)]

        result.append(
            RuleSpec(
                sutra_id=f"{template.sutra_id}.__synthetic_{index:06d}",
                text=template.text,
                kind=template.kind,
                conditions=list(template.conditions),
                transformations=list(template.transformations),
                adhikara=template.adhikara,
                anuvrtti=template.anuvrtti,
                priority=template.priority,
                tags=template.tags,
                metadata={
                    **template.metadata,
                    "synthetic": True,
                    "template": template.sutra_id,
                },
            )
        )

    return result


# ============================================================================
# 10. Self-test
# ============================================================================

def self_test() -> None:
    compiler = build_scaled_prototype_compiler()

    assert compiler.compiled
    assert compiler.report.success
    assert compiler.report.compiled_rules == 5

    summary = compiler.index.summary()

    assert summary["rules"] == 5
    assert summary["root_keys"] >= 3
    assert summary["tag_keys"] >= 5

    # Candidate retrieval should avoid scanning the whole corpus conceptually.
    candidates = compiler.candidate_rules(root="bhū")
    ids = {item.spec.sutra_id for item in candidates}

    assert "scaled.guna.bhu" in ids

    # Condition resolution.
    context = compiler.build_context(
        root="bhū",
        suffix="ti",
        stem="bhū",
    )

    ranked = compiler.resolve(
        context,
        root="bhū",
    )

    assert ranked
    assert ranked[0].rule.spec.sutra_id == "scaled.guna.bhu"

    # Adhikāra indexing.
    adhikara_rules = compiler.candidate_rules(
        adhikara="scaled-guna"
    )
    assert len(adhikara_rules) >= 2

    # Tag indexing.
    gam_rules = compiler.candidate_rules(tags=("gam",))
    assert any(
        rule.spec.sutra_id == "scaled.gam.present"
        for rule in gam_rules
    )

    # Manifest must be serializable.
    payload = compiler.export_json()
    assert "scaled.guna.bhu" in payload
    assert "scaled.gam.present" in payload

    # Large-corpus construction test.
    synthetic = replicate_rules(
        prototype_scaled_rules(),
        1000,
    )

    large = ScaledPaniniCompiler(
        config=CompilerConfig(
            max_candidates=64,
            strict_validation=True,
        )
    )
    large.add_rules(synthetic)
    report = large.compile()

    assert report.success
    assert report.compiled_rules == 1000

    # Candidate retrieval must be bounded.
    candidates = large.candidate_rules(
        root="bhū",
        limit=64,
    )

    assert len(candidates) <= 64


# ============================================================================
# 11. Demonstration
# ============================================================================

def demo() -> None:
    compiler = build_scaled_prototype_compiler()

    print("=" * 80)
    print("PANINI LANGUAGE MACHINE — SCALED AṢṬĀDHYĀYĪ COMPILER")
    print("=" * 80)

    print("\nCompilation report:")
    for key, value in compiler.report.to_dict().items():
        if key != "diagnostics":
            print(f"  {key:<20}: {value}")

    print("\nIndex:")
    for key, value in compiler.index.summary().items():
        print(f"  {key:<20}: {value}")

    print("\nCandidate retrieval:")

    for root in ("bhū", "gam", "nī"):
        candidates = compiler.candidate_rules(
            root=root,
            limit=10,
        )

        print(f"\n  root={root}")

        for candidate in candidates:
            print(
                f"    {candidate.spec.sutra_id:<35} "
                f"priority={candidate.spec.priority}"
            )

    print("\nConflict resolution:")

    context = compiler.build_context(
        root="gam",
        suffix="ti",
        stem="gam",
    )

    ranked = compiler.resolve(
        context,
        root="gam",
    )

    for item in ranked[:5]:
        print(
            f"  {item.rule.spec.sutra_id:<35} "
            f"score={item.score}"
        )

    print("\n1000-rule compilation test:")

    synthetic = replicate_rules(
        prototype_scaled_rules(),
        1000,
    )

    large = ScaledPaniniCompiler()
    large.add_rules(synthetic)
    report = large.compile()

    print(
        f"  compiled={report.compiled_rules}, "
        f"errors={report.errors}, "
        f"warnings={report.warnings}"
    )

    print("\nSelf-test:")
    self_test()
    print("  PASS")


if __name__ == "__main__":
    demo()
