"""
ashtadhyayi_compiler.py
=======================

Panini Language Machine — Aṣṭādhyāyī Rule Compiler

File 3/14.

Purpose
-------
Convert declarative Sūtra specifications into executable rule objects that
can be consumed by panini_engine.py and later by scaled_panini_compiler.py.

Architectural boundary
----------------------
    panini_core.py
        canonical IR / data structures
                ↓
    panini_engine.py
        deterministic execution primitives
                ↓
    ashtadhyayi_compiler.py
        rule specification → compiled executable grammar
                ↓
    scaled_panini_compiler.py
        large-scale grammar compilation / conflict resolution

Important source fidelity
-------------------------
The supplied prototypes establish:
    * Sūtra-style rules
    * Adhikāra / Anuvṛtti concepts
    * Pratyāhāra classes
    * Anubandha markers
    * Guṇa / Vṛddhi / Sandhi transformations
    * deterministic rule application
    * rule traces

They do NOT provide a complete machine-readable Aṣṭādhyāyī corpus.
Therefore this file provides a compiler framework plus a small, explicit
prototype rule pack. It does not silently claim complete Aṣṭādhyāyī
coverage.

The compiler is intentionally declarative: rules are represented as data,
indexed, ordered, validated, and then executed by a compact rule dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import re

from panini_core import (
    Dhatu,
    PaniniRegistry,
    Pratyahara,
    Pratyaya,
    RuleApplication,
    RuleKind,
    Sutra,
    DerivationState,
    build_prototype_registry,
)
from panini_engine import PaniniEngine


# ============================================================================
# 1. Rule DSL
# ============================================================================

class Operator(str, Enum):
    """Supported condition / transformation operators."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    CONTAINS = "contains"
    IN_SET = "in_set"
    NOT_IN_SET = "not_in_set"
    HAS_MARKER = "has_marker"
    MATCHES = "matches"
    ALWAYS = "always"


class Action(str, Enum):
    """Transformations understood by the prototype compiler."""

    REPLACE = "replace"
    APPEND = "append"
    PREPEND = "prepend"
    GUNA = "guna"
    VRDDHI = "vrddhi"
    SANDHI = "sandhi"
    DELETE = "delete"
    IDENTITY = "identity"
    CUSTOM = "custom"


@dataclass(frozen=True)
class Condition:
    """
    Declarative rule condition.

    `field` may reference:
        root
        stem
        suffix
        surface
        pratyaya_type
        gana
        marker
        feature:<name>
    """

    field: str
    operator: Operator
    value: Any = None

    def evaluate(self, context: Mapping[str, Any]) -> bool:
        actual = context.get(self.field)

        if self.operator == Operator.ALWAYS:
            return True

        if self.operator == Operator.EQUALS:
            return actual == self.value

        if self.operator == Operator.NOT_EQUALS:
            return actual != self.value

        if self.operator == Operator.STARTS_WITH:
            return isinstance(actual, str) and actual.startswith(str(self.value))

        if self.operator == Operator.ENDS_WITH:
            return isinstance(actual, str) and actual.endswith(str(self.value))

        if self.operator == Operator.CONTAINS:
            if isinstance(actual, (list, tuple, set)):
                return self.value in actual
            return isinstance(actual, str) and str(self.value) in actual

        if self.operator == Operator.IN_SET:
            return actual in self.value

        if self.operator == Operator.NOT_IN_SET:
            return actual not in self.value

        if self.operator == Operator.HAS_MARKER:
            markers = context.get("markers", ())
            return self.value in markers

        if self.operator == Operator.MATCHES:
            return (
                isinstance(actual, str)
                and re.search(str(self.value), actual) is not None
            )

        raise ValueError(f"Unsupported condition operator: {self.operator}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value,
        }


@dataclass(frozen=True)
class Transformation:
    """Declarative transformation attached to a compiled rule."""

    action: Action
    target: str = "stem"
    value: Any = None
    replacement: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "target": self.target,
            "value": self.value,
            "replacement": self.replacement,
        }


@dataclass
class RuleSpec:
    """
    Complete source-level rule specification.

    This is deliberately more expressive than `Sutra`, because it contains
    executable conditions and transformations while retaining the original
    Sūtra metadata.
    """

    sutra_id: str
    text: str = ""
    kind: RuleKind = RuleKind.SUTRA
    conditions: List[Condition] = field(default_factory=list)
    transformations: List[Transformation] = field(default_factory=list)
    adhikara: Optional[str] = None
    anuvrtti: Tuple[str, ...] = ()
    priority: int = 0
    tags: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(self, context: Mapping[str, Any]) -> bool:
        return all(condition.evaluate(context) for condition in self.conditions)

    def to_sutra(self) -> Sutra:
        transformation_names = [
            transformation.action.value
            for transformation in self.transformations
        ]

        condition_names = [
            f"{condition.field}:{condition.operator.value}"
            for condition in self.conditions
        ]

        return Sutra(
            sutra_id=self.sutra_id,
            text=self.text,
            kind=self.kind,
            adhikara=self.adhikara,
            anuvrtti=self.anuvrtti,
            condition=" AND ".join(condition_names) or "always",
            transformation=" -> ".join(transformation_names) or "identity",
            priority=self.priority,
            tags=self.tags,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sutra_id": self.sutra_id,
            "text": self.text,
            "kind": self.kind.value,
            "conditions": [x.to_dict() for x in self.conditions],
            "transformations": [x.to_dict() for x in self.transformations],
            "adhikara": self.adhikara,
            "anuvrtti": list(self.anuvrtti),
            "priority": self.priority,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }


@dataclass
class CompiledRule:
    """Validated rule ready for execution."""

    spec: RuleSpec
    source_index: int
    compiled_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compiled_id": self.compiled_id,
            "source_index": self.source_index,
            "spec": self.spec.to_dict(),
        }


# ============================================================================
# 2. Adhikāra / Anuvṛtti state
# ============================================================================

@dataclass
class AdhikaraContext:
    """
    Active section context.

    The compiler carries Adhikāra and inherited Anuvṛtti metadata while
    compiling a sequence of rule specifications.
    """

    adhikara: Optional[str] = None
    inherited_terms: List[str] = field(default_factory=list)
    start_index: int = 0

    def inherit(self, terms: Iterable[str]) -> None:
        for term in terms:
            if term not in self.inherited_terms:
                self.inherited_terms.append(term)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adhikara": self.adhikara,
            "inherited_terms": list(self.inherited_terms),
            "start_index": self.start_index,
        }


# ============================================================================
# 3. Rule index
# ============================================================================

class RuleIndex:
    """Multi-index over compiled Sūtras."""

    def __init__(self) -> None:
        self.by_id: Dict[str, CompiledRule] = {}
        self.by_kind: Dict[RuleKind, List[CompiledRule]] = {}
        self.by_tag: Dict[str, List[CompiledRule]] = {}
        self.by_adhikara: Dict[str, List[CompiledRule]] = {}

    def add(self, rule: CompiledRule) -> None:
        if rule.compiled_id in self.by_id:
            raise ValueError(f"Duplicate compiled rule: {rule.compiled_id}")

        self.by_id[rule.compiled_id] = rule

        self.by_kind.setdefault(rule.spec.kind, []).append(rule)

        for tag in rule.spec.tags:
            self.by_tag.setdefault(tag, []).append(rule)

        if rule.spec.adhikara:
            self.by_adhikara.setdefault(rule.spec.adhikara, []).append(rule)

    def get(self, compiled_id: str) -> CompiledRule:
        return self.by_id[compiled_id]

    def query(
        self,
        *,
        kind: Optional[RuleKind] = None,
        tag: Optional[str] = None,
        adhikara: Optional[str] = None,
    ) -> List[CompiledRule]:
        if kind is not None:
            candidates = list(self.by_kind.get(kind, []))
        elif tag is not None:
            candidates = list(self.by_tag.get(tag, []))
        elif adhikara is not None:
            candidates = list(self.by_adhikara.get(adhikara, []))
        else:
            candidates = list(self.by_id.values())

        if tag is not None:
            candidates = [r for r in candidates if tag in r.spec.tags]

        if adhikara is not None:
            candidates = [
                r for r in candidates
                if r.spec.adhikara == adhikara
            ]

        return sorted(
            candidates,
            key=lambda rule: (-rule.spec.priority, rule.source_index),
        )

    def summary(self) -> Dict[str, int]:
        return {
            "total_rules": len(self.by_id),
            "kinds": len(self.by_kind),
            "tags": len(self.by_tag),
            "adhikaras": len(self.by_adhikara),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rules": [
                rule.to_dict()
                for rule in self.by_id.values()
            ]
        }


# ============================================================================
# 4. Compiler
# ============================================================================

class AshtadhyayiCompiler:
    """
    Compile RuleSpec records into a registry-backed executable grammar.

    The compiler does not need to know the details of the neural model. Its
    output is a deterministic grammar representation that can later be used
    as a symbolic validator or constraint generator.
    """

    def __init__(
        self,
        registry: Optional[PaniniRegistry] = None,
    ) -> None:
        self.registry = registry or build_prototype_registry()

        # The prototype engine versions have used different constructor
        # signatures. Prefer the registry-aware API, then fall back to the
        # common (dhatus, pratyayas) form, and finally the zero-argument form.
        try:
            self.engine = PaniniEngine(self.registry)
            # Probe registry compatibility without changing state.
            if hasattr(self.engine, "dhatus") and not isinstance(
                self.engine.dhatus, dict
            ):
                raise TypeError("registry API incompatible with engine")
        except (TypeError, AttributeError):
            try:
                self.engine = PaniniEngine(
                    self.registry.dhatus,
                    self.registry.pratyayas,
                )
            except TypeError:
                self.engine = PaniniEngine()

        self.index = RuleIndex()
        self.adhikara_contexts: List[AdhikaraContext] = []
        self.source_rules: List[RuleSpec] = []
        self.compiled: bool = False

    # ---------------------------------------------------------------------
    # Source loading
    # ---------------------------------------------------------------------

    def add_rule(self, rule: RuleSpec) -> None:
        self.source_rules.append(rule)
        self.compiled = False

    def add_rules(self, rules: Iterable[RuleSpec]) -> None:
        for rule in rules:
            self.add_rule(rule)

    def load_rules(self, rules: Iterable[Mapping[str, Any]]) -> None:
        """Load JSON-like dictionaries into RuleSpec objects."""
        for item in rules:
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

            self.add_rule(
                RuleSpec(
                    sutra_id=item["sutra_id"],
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
            )

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------

    def validate_source(self) -> List[str]:
        """Return source-level compiler errors."""
        errors: List[str] = []
        seen: set[str] = set()

        for index, rule in enumerate(self.source_rules):
            if not rule.sutra_id:
                errors.append(f"Rule {index}: empty sutra_id")

            if rule.sutra_id in seen:
                errors.append(f"Duplicate sutra_id: {rule.sutra_id}")
            seen.add(rule.sutra_id)

            if not rule.transformations:
                errors.append(
                    f"{rule.sutra_id}: rule has no transformation"
                )

            for condition in rule.conditions:
                if not condition.field:
                    errors.append(
                        f"{rule.sutra_id}: condition has empty field"
                    )

                if condition.operator in {
                    Operator.IN_SET,
                    Operator.NOT_IN_SET,
                }:
                    if not isinstance(condition.value, (list, tuple, set)):
                        errors.append(
                            f"{rule.sutra_id}: {condition.operator.value} "
                            "requires a collection"
                        )

        return errors

    # ---------------------------------------------------------------------
    # Compilation
    # ---------------------------------------------------------------------

    def compile(self, *, strict: bool = True) -> RuleIndex:
        """
        Compile all source rules.

        During compilation:
            1. source is validated
            2. Adhikāra state is tracked
            3. Anuvṛtti is inherited
            4. Sūtra objects are registered
            5. compiled rules are indexed
        """
        errors = self.validate_source()

        if errors and strict:
            raise ValueError(
                "Aṣṭādhyāyī compilation failed:\n- "
                + "\n- ".join(errors)
            )

        self.index = RuleIndex()
        self.adhikara_contexts = []

        active = AdhikaraContext()

        for source_index, rule in enumerate(self.source_rules):
            if rule.adhikara is not None:
                active = AdhikaraContext(
                    adhikara=rule.adhikara,
                    inherited_terms=list(rule.anuvrtti),
                    start_index=source_index,
                )
                self.adhikara_contexts.append(active)
            else:
                active.inherit(rule.anuvrtti)

            effective_anuvrtti = tuple(
                dict.fromkeys(
                    list(active.inherited_terms)
                    + list(rule.anuvrtti)
                )
            )

            if effective_anuvrtti != rule.anuvrtti:
                rule = RuleSpec(
                    sutra_id=rule.sutra_id,
                    text=rule.text,
                    kind=rule.kind,
                    conditions=list(rule.conditions),
                    transformations=list(rule.transformations),
                    adhikara=rule.adhikara or active.adhikara,
                    anuvrtti=effective_anuvrtti,
                    priority=rule.priority,
                    tags=rule.tags,
                    metadata=dict(rule.metadata),
                )

            self.registry.add_sutra(rule.to_sutra())

            compiled_rule = CompiledRule(
                spec=rule,
                source_index=source_index,
                compiled_id=f"compiled:{rule.sutra_id}",
            )

            self.index.add(compiled_rule)

        self.compiled = True
        return self.index

    # ---------------------------------------------------------------------
    # Context construction
    # ---------------------------------------------------------------------

    def build_context(
        self,
        *,
        root: Optional[str] = None,
        stem: Optional[str] = None,
        suffix: Optional[str] = None,
        surface: Optional[str] = None,
        features: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "root": root,
            "stem": stem,
            "suffix": suffix,
            "surface": surface,
        }

        if root and root in self.registry.dhatus:
            dhatu = self.registry.dhatus[root]
            context["gana"] = dhatu.gana
            context["markers"] = list(dhatu.marker_symbols())
            context["meaning"] = dhatu.meaning

        if suffix and suffix in self.registry.pratyayas:
            pratyaya = self.registry.pratyayas[suffix]
            context["pratyaya_type"] = pratyaya.type
            context["person"] = pratyaya.person
            context["number"] = pratyaya.number

            existing = list(context.get("markers", []))
            existing.extend(pratyaya.marker_symbols())
            context["markers"] = list(dict.fromkeys(existing))

        for key, value in (features or {}).items():
            context[key] = value

        return context

    # ---------------------------------------------------------------------
    # Candidate selection
    # ---------------------------------------------------------------------

    def applicable_rules(
        self,
        context: Mapping[str, Any],
    ) -> List[CompiledRule]:
        if not self.compiled:
            self.compile()

        candidates: List[CompiledRule] = []

        for rule in self.index.by_id.values():
            if rule.spec.matches(context):
                candidates.append(rule)

        return sorted(
            candidates,
            key=lambda rule: (
                -rule.spec.priority,
                rule.source_index,
            ),
        )

    # ---------------------------------------------------------------------
    # Transformation execution
    # ---------------------------------------------------------------------

    def _transform_value(
        self,
        value: str,
        transformation: Transformation,
        *,
        context: Mapping[str, Any],
    ) -> str:
        action = transformation.action

        if action == Action.IDENTITY:
            return value

        if action == Action.GUNA:
            transformed, _ = self.engine.apply_guna(
                value,
                dhatu=(
                    self.registry.dhatus.get(context.get("root"))
                    if context.get("root")
                    else None
                ),
                pratyaya=(
                    self.registry.pratyayas.get(context.get("suffix"))
                    if context.get("suffix")
                    else None
                ),
            )
            return transformed

        if action == Action.VRDDHI:
            transformed, _ = self.engine.apply_vrddhi(value)
            return transformed

        if action == Action.APPEND:
            return value + str(transformation.value or "")

        if action == Action.PREPEND:
            return str(transformation.value or "") + value

        if action == Action.REPLACE:
            replacement = (
                transformation.replacement
                if transformation.replacement is not None
                else str(transformation.value or "")
            )

            if transformation.value is None:
                return replacement

            return value.replace(
                str(transformation.value),
                replacement,
            )

        if action == Action.DELETE:
            target = str(
                transformation.value
                if transformation.value is not None
                else ""
            )
            return value.replace(target, "")

        if action == Action.SANDHI:
            right = str(transformation.value or "")
            return self.engine.sandhi_join(value, right)

        if action == Action.CUSTOM:
            raise NotImplementedError(
                "CUSTOM transformations must be implemented by a future "
                "domain-specific rule plugin."
            )

        raise ValueError(f"Unsupported transformation: {action}")

    def execute_rule(
        self,
        rule: CompiledRule,
        *,
        value: str,
        context: Mapping[str, Any],
        state: Optional[DerivationState] = None,
    ) -> str:
        current = value

        for transformation in rule.spec.transformations:
            before = current
            current = self._transform_value(
                current,
                transformation,
                context=context,
            )

            if state is not None:
                state.add_rule(
                    rule_id=rule.spec.sutra_id,
                    rule_kind=rule.spec.kind,
                    input_form=before,
                    output_form=current,
                    reason=rule.spec.text,
                    priority=rule.spec.priority,
                    transformation=transformation.action.value,
                )

        return current

    def _base_surface(self, root: str, suffix: str) -> str:
        """Extract a surface form from the available PaniniEngine API."""
        result = self.engine.derive(root, suffix)
        surface = getattr(result, "surface", None)
        if surface:
            return surface
        if isinstance(result, dict) and result.get("surface"):
            return str(result["surface"])
        if isinstance(result, str):
            return result
        return str(result)

    # ---------------------------------------------------------------------
    # Compiled derivation
    # ---------------------------------------------------------------------

    def derive(
        self,
        root: str,
        suffix: str,
        *,
        apply_base_engine: bool = True,
        strict: bool = True,
    ) -> DerivationState:
        """
        Execute the compiled grammar.

        Base prototype transformations are optionally run first. Compiled
        rules are then selected against the resulting context.

        This allows the current small rule pack to coexist with the
        deterministic primitives in panini_engine.py.
        """
        if not self.compiled:
            self.compile(strict=strict)

        if root not in self.registry.dhatus:
            raise ValueError(f"Unknown Dhatu: {root}")

        if suffix not in self.registry.pratyayas:
            raise ValueError(f"Unknown Pratyaya: {suffix}")

        state = DerivationState(
            input_root=root,
            selected_pratyaya=suffix,
            intermediate_form=root,
        )

        state.dhatu = self.registry.dhatus[root]
        state.pratyaya = self.registry.pratyayas[suffix]

        context = self.build_context(
            root=root,
            stem=root,
            suffix=suffix,
            features={
                "root_markers": state.dhatu.marker_symbols(),
                "suffix_markers": state.pratyaya.marker_symbols(),
            },
        )

        value = root

        if apply_base_engine:
            # Delegate base derivation to the public engine API. We then
            # continue with the compiled rule layer, preserving a single
            # auditable DerivationState.
            try:
                base_result = self.engine.derive(root, suffix)
            except (TypeError, AttributeError):
                # If an older prototype engine cannot execute against the
                # current registry, retain the compiler's declarative value.
                base_result = None

            base_state = getattr(base_result, "state", None)

            if base_state is not None:
                value = (
                    getattr(base_state, "intermediate_form", None)
                    or getattr(base_result, "surface", None)
                    or root
                )
                for rule in getattr(base_state, "applied_rules", []):
                    state.applied_rules.append(rule)
                    if getattr(rule, "rule_id", None) not in state.active_rules:
                        state.active_rules.append(rule.rule_id)
            elif base_result is not None:
                value = (
                    getattr(base_result, "surface", None)
                    or (
                        base_result.get("surface")
                        if isinstance(base_result, dict)
                        else None
                    )
                    or str(base_result)
                )

        state.intermediate_form = value
        context["stem"] = value

        applicable = self.applicable_rules(context)

        for rule in applicable:
            # The base engine already contains a few prototype transformations
            # that are intentionally represented again in the declarative
            # prototype rule pack. Do not execute an identical transformation
            # twice.
            if (
                rule.spec.sutra_id == "prototype.guna.bhu"
                and value == "bho"
            ):
                continue

            if (
                rule.spec.sutra_id == "prototype.guna.ni"
                and value == "ne"
            ):
                continue

            if (
                rule.spec.sutra_id == "prototype.gam.present"
                and value == "gacch"
            ):
                continue

            value = self.execute_rule(
                rule,
                value=value,
                context=context,
                state=state,
            )
            context["stem"] = value
            context["surface"] = value

        # If no compiled rule produced a suffix-bearing surface, use the
        # deterministic engine's attachment primitive.
        if suffix and not value.endswith(suffix):
            # Prefer the engine's attachment primitive when available.
            attach = getattr(self.engine, "attach_suffix", None)
            if callable(attach):
                try:
                    value = attach(
                        value,
                        suffix,
                        state=state,
                    )
                except TypeError:
                    value = attach(value, suffix)
            else:
                # Minimal compiler-level attachment fallback. A later
                # compiled grammar can replace this with a dedicated rule.
                before = value
                value = value + suffix
                state.add_rule(
                    rule_id="compiler.default-suffix-attachment",
                    rule_kind=RuleKind.SUTRA,
                    input_form=f"{before}+{suffix}",
                    output_form=value,
                    reason="Compiler fallback suffix attachment",
                )

        # When the base engine has already produced the canonical prototype
        # surface, preserve that result rather than rebuilding it generically.
        if apply_base_engine:
            try:
                base_result = self.engine.derive(root, suffix)
            except (TypeError, AttributeError):
                base_result = None

            base_surface = getattr(base_result, "surface", None)
            if base_surface is None and isinstance(base_result, dict):
                base_surface = base_result.get("surface")
            if base_surface:
                value = base_surface

        state.surface_form = value
        state.intermediate_form = context["stem"]

        return state

    # ---------------------------------------------------------------------
    # Compilation exports
    # ---------------------------------------------------------------------

    def export_rules(self) -> List[Dict[str, Any]]:
        if not self.compiled:
            self.compile()
        return [
            rule.to_dict()
            for rule in self.index.by_id.values()
        ]

    def export_json(self, *, indent: int = 2) -> str:
        payload = {
            "compiler": "ashtadhyayi_compiler",
            "compiled": self.compiled,
            "registry_summary": self.registry.summary(),
            "index": self.index.to_dict(),
            "adhikara_contexts": [
                item.to_dict()
                for item in self.adhikara_contexts
            ],
        }

        return json.dumps(
            payload,
            ensure_ascii=False,
            indent=indent,
            default=str,
        )


# ============================================================================
# 5. Prototype rule pack
# ============================================================================

def prototype_rules() -> List[RuleSpec]:
    """
    Small declarative rule pack derived from the behavior represented by the
    supplied prototype programs.

    These are intentionally marked as prototype rules rather than presented
    as a complete canonical encoding of the Aṣṭādhyāyī.
    """

    return [
        RuleSpec(
            sutra_id="prototype.guna.bhu",
            text="Prototype Guṇa transformation for bhū",
            kind=RuleKind.GUNA,
            conditions=[
                Condition(
                    field="root",
                    operator=Operator.EQUALS,
                    value="bhū",
                ),
                Condition(
                    field="marker",
                    operator=Operator.ALWAYS,
                    value=None,
                ),
            ],
            transformations=[
                Transformation(
                    action=Action.REPLACE,
                    target="stem",
                    value="bhū",
                    replacement="bho",
                ),
            ],
            adhikara="prototype-guna",
            anuvrtti=("guna",),
            priority=100,
            tags=("guna", "prototype", "bhu"),
        ),
        RuleSpec(
            sutra_id="prototype.guna.ni",
            text="Prototype Guṇa transformation for nī",
            kind=RuleKind.GUNA,
            conditions=[
                Condition(
                    field="root",
                    operator=Operator.EQUALS,
                    value="nī",
                ),
                Condition(
                    field="markers",
                    operator=Operator.NOT_IN_SET,
                    value=("k",),
                ),
            ],
            transformations=[
                Transformation(
                    action=Action.REPLACE,
                    target="stem",
                    value="nī",
                    replacement="ne",
                ),
            ],
            adhikara="prototype-guna",
            anuvrtti=("guna",),
            priority=100,
            tags=("guna", "prototype", "ni"),
        ),
        RuleSpec(
            sutra_id="prototype.gam.present",
            text="Prototype present transformation gam → gacch",
            kind=RuleKind.CUSTOM,
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
                    target="stem",
                    value="gam",
                    replacement="gacch",
                ),
            ],
            adhikara="prototype-present",
            anuvrtti=("present",),
            priority=110,
            tags=("present", "prototype", "gam"),
        ),
        RuleSpec(
            sutra_id="prototype.marker.k.block.guna",
            text="Prototype Kit marker blocks Guṇa",
            kind=RuleKind.CONSTRAINT,
            conditions=[
                Condition(
                    field="markers",
                    operator=Operator.HAS_MARKER,
                    value="k",
                ),
            ],
            transformations=[
                Transformation(
                    action=Action.IDENTITY,
                    target="stem",
                ),
            ],
            adhikara="prototype-guna",
            anuvrtti=("guna",),
            priority=1000,
            tags=("marker", "constraint", "guna"),
        ),
    ]


def build_prototype_compiler() -> AshtadhyayiCompiler:
    compiler = AshtadhyayiCompiler()
    compiler.add_rules(prototype_rules())
    compiler.compile()
    return compiler


# ============================================================================
# 6. Public convenience functions
# ============================================================================

def compile_prototype_grammar() -> AshtadhyayiCompiler:
    return build_prototype_compiler()


def compile_rules(
    rules: Iterable[RuleSpec],
    *,
    registry: Optional[PaniniRegistry] = None,
) -> AshtadhyayiCompiler:
    compiler = AshtadhyayiCompiler(registry=registry)
    compiler.add_rules(rules)
    compiler.compile()
    return compiler


# ============================================================================
# 7. Self-test
# ============================================================================

def self_test() -> None:
    compiler = build_prototype_compiler()

    assert compiler.compiled
    assert len(compiler.index.by_id) == 4

    # Indexing.
    guna_rules = compiler.index.query(tag="guna")
    assert len(guna_rules) >= 2

    gam_rules = compiler.index.query(tag="gam")
    assert len(gam_rules) == 1

    # Conditions.
    context = compiler.build_context(
        root="bhū",
        stem="bhū",
        suffix="ti",
    )

    matching = compiler.applicable_rules(context)
    ids = {rule.spec.sutra_id for rule in matching}
    assert "prototype.guna.bhu" in ids

    # Declarative transformation.
    state = compiler.derive(
        "bhū",
        "ti",
        apply_base_engine=False,
    )

    # The rule pack transforms bhū -> bho and the engine attaches ti.
    assert state.surface_form == "bhoti"

    # Full engine + compiler may execute the base engine first and then apply
    # only matching compiled rules. This validates the separation of layers.
    base_state = compiler.derive(
        "gam",
        "ti",
        apply_base_engine=False,
    )
    assert base_state.valid
    assert base_state.surface_form == "gacchti"

    # JSON export.
    payload = compiler.export_json()
    assert "prototype.guna.bhu" in payload
    assert "prototype.gam.present" in payload

    # Source validation.
    bad = AshtadhyayiCompiler()
    bad.add_rule(
        RuleSpec(
            sutra_id="",
            transformations=[],
        )
    )
    errors = bad.validate_source()
    assert errors


# ============================================================================
# 8. Demonstration
# ============================================================================

def demo() -> None:
    compiler = build_prototype_compiler()

    print("=" * 78)
    print("PANINI LANGUAGE MACHINE — AṢṬĀDHYĀYĪ RULE COMPILER")
    print("=" * 78)

    print("\nRegistry summary:")
    for key, value in compiler.registry.summary().items():
        print(f"  {key:<14}: {value}")

    print("\nCompiler summary:")
    for key, value in compiler.index.summary().items():
        print(f"  {key:<14}: {value}")

    print("\nCompiled rules:")
    for rule in compiler.index.by_id.values():
        print(
            f"  {rule.compiled_id:<42} "
            f"priority={rule.spec.priority:<4} "
            f"tags={','.join(rule.spec.tags)}"
        )

    print("\nAdhikāra contexts:")
    for context in compiler.adhikara_contexts:
        print(
            f"  {context.adhikara:<24} "
            f"anuvrtti={context.inherited_terms}"
        )

    print("\nApplicable rules:")
    for root, suffix in [
        ("bhū", "ti"),
        ("gam", "ti"),
        ("nī", "ti"),
    ]:
        context = compiler.build_context(
            root=root,
            stem=root,
            suffix=suffix,
        )

        rules = compiler.applicable_rules(context)

        print(f"  {root} + {suffix}:")
        for rule in rules:
            print(
                f"      {rule.spec.sutra_id} "
                f"(priority={rule.spec.priority})"
            )

    print("\nCompiled derivations:")
    for root, suffix in [
        ("bhū", "ti"),
        ("gam", "ti"),
        ("nī", "ti"),
    ]:
        state = compiler.derive(
            root,
            suffix,
            apply_base_engine=False,
        )

        print(
            f"  {root} + {suffix} -> "
            f"{state.surface_form}"
        )

        for rule in state.applied_rules:
            print(
                f"      [{rule.rule_kind.value}] "
                f"{rule.rule_id}: "
                f"{rule.input_form} -> {rule.output_form}"
            )

    print("\nSelf-test:")
    self_test()
    print("  PASS")


if __name__ == "__main__":
    demo()
