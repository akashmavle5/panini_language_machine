"""
paninian_english_llm.py
=======================

Panini Language Machine — File 10/14

English-side Vyakarana Graph Automata (VGA) prototype.

Purpose
-------
Transform a linear English sentence into a non-linear Paninian-style semantic
frame before a neural language model consumes/generates tokens.

Pipeline:
    text -> structural atoms -> action engine -> valency routing
         -> Kāraka graph -> generation constraints

This is a research prototype, not a complete English parser or LLM.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Atom:
    token: str
    type: str
    category: str
    index: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Binding:
    role: str
    token: Optional[str]
    token_index: Optional[int]
    confidence: float
    evidence: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# VGA engine
# ---------------------------------------------------------------------------

class VyakaranaGraphAutomata:
    """Lightweight Paninian structural engine for English."""

    def __init__(self) -> None:
        # Dual lexicon: roots/action engines and functional markers.
        self.root_lexicon = {
            "merchant": "Noun",
            "knife": "Noun",
            "apple": "Noun",
            "orchard": "Noun",
            "book": "Noun",
            "student": "Noun",
            "teacher": "Noun",
            "ram": "Noun",
            "sita": "Noun",
            "cut": "Verb_Engine",
            "slice": "Verb_Engine",
            "read": "Verb_Engine",
            "reads": "Verb_Engine",
            "give": "Verb_Engine",
            "gives": "Verb_Engine",
            "build": "Verb_Engine",
            "builds": "Verb_Engine",
            "write": "Verb_Engine",
            "writes": "Verb_Engine",
        }

        # Marker -> semantic role.  The previous implementation stored a
        # marker class but compared it with the literal marker; this version
        # keeps both values explicit and therefore avoids that inconsistency.
        self.marker_lexicon = {
            "by": ("Karta_Marker", "Kartā"),
            "with": ("Karana_Marker", "Karaṇa"),
            "to": ("Sampradana_Marker", "Sampradāna"),
            "from": ("Apadana_Marker", "Apādāna"),
            "of": ("Sambandha_Marker", "Sambandha"),
            "in": ("Adhikarana_Marker", "Adhikaraṇa"),
            "on": ("Adhikarana_Marker", "Adhikaraṇa"),
            "at": ("Adhikarana_Marker", "Adhikaraṇa"),
            "the": ("Article", None),
            "a": ("Article", None),
            "an": ("Article", None),
        }

        self.valency_demands = {
            "cut": {"required": ["Kartā", "Karma"],
                    "optional": ["Karaṇa", "Adhikaraṇa"]},
            "slice": {"required": ["Kartā", "Karma"],
                      "optional": ["Karaṇa", "Adhikaraṇa"]},
            "read": {"required": ["Kartā", "Karma"],
                     "optional": ["Adhikaraṇa"]},
            "reads": {"required": ["Kartā", "Karma"],
                      "optional": ["Adhikaraṇa"]},
            "give": {"required": ["Kartā", "Karma", "Sampradāna"],
                     "optional": ["Karaṇa"]},
            "gives": {"required": ["Kartā", "Karma", "Sampradāna"],
                      "optional": ["Karaṇa"]},
            "build": {"required": ["Kartā", "Karma"],
                      "optional": ["Karaṇa", "Adhikaraṇa"]},
            "builds": {"required": ["Kartā", "Karma"],
                       "optional": ["Karaṇa", "Adhikaraṇa"]},
            "write": {"required": ["Kartā", "Karma"],
                      "optional": ["Karaṇa", "Adhikaraṇa"]},
            "writes": {"required": ["Kartā", "Karma"],
                       "optional": ["Karaṇa", "Adhikaraṇa"]},
        }

        self.modifier_words = {
            "swift", "swiftly", "ripe", "young", "old", "new", "big",
            "small", "red", "green", "large", "smart", "fast", "good",
            "beautiful", "semantic", "symbolic", "neural",
        }

    # ------------------------------------------------------------------
    # Lexical / atom layer
    # ------------------------------------------------------------------

    def parse_to_atoms(self, sentence_string: str) -> List[Atom]:
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", sentence_string.lower())
        atoms: List[Atom] = []

        for index, word in enumerate(words):
            if word in self.root_lexicon:
                category = self.root_lexicon[word]
                atoms.append(Atom(word, "ROOT", category, index))
            elif word in self.marker_lexicon:
                marker_class, _ = self.marker_lexicon[word]
                atoms.append(Atom(word, "MARKER", marker_class, index))
            elif word in self.modifier_words or word.endswith("ly"):
                atoms.append(Atom(word, "MODIFIER", "Modifier", index))
            else:
                # Unknown words remain explicit instead of being silently
                # discarded; later lexical expansion can classify them.
                atoms.append(Atom(word, "UNKNOWN_ROOT", "NounOrModifier", index))

        return atoms

    # ------------------------------------------------------------------
    # Verb / valency layer
    # ------------------------------------------------------------------

    def find_action_engine(self, atoms: Sequence[Atom]) -> Atom:
        verb = next(
            (atom for atom in atoms if atom.category == "Verb_Engine"),
            None,
        )
        if verb is None:
            raise ValueError(
                "Structural Failure: No Action Engine (verb) detected."
            )
        return verb

    def get_valency(self, verb: Atom) -> Dict[str, List[str]]:
        return self.valency_demands.get(
            verb.token,
            {"required": ["Kartā"], "optional": []},
        )

    # ------------------------------------------------------------------
    # Kāraka routing
    # ------------------------------------------------------------------

    def execute_valency_routing(self, atom_set: Sequence[Atom]) -> Dict[str, Any]:
        verb_atom = self.find_action_engine(atom_set)
        valency = self.get_valency(verb_atom)

        bindings: Dict[str, Binding] = {}
        marker_role_by_index: Dict[int, str] = {}

        # Identify marker -> following noun span.
        for idx, atom in enumerate(atom_set):
            if atom.type != "MARKER":
                continue
            marker_info = self.marker_lexicon.get(atom.token)
            if marker_info is None:
                continue
            _, role = marker_info
            if role is None:
                continue

            # Find the first noun-like atom after the marker, allowing
            # articles/modifiers in between.
            for candidate in atom_set[idx + 1: idx + 5]:
                if candidate.category in {"Noun", "NounOrModifier"}:
                    marker_role_by_index[candidate.index] = role
                    break

        # Strong marker-based bindings first.
        for atom in atom_set:
            role = marker_role_by_index.get(atom.index)
            if role is None:
                continue
            if role not in bindings:
                bindings[role] = Binding(
                    role=role,
                    token=atom.token,
                    token_index=atom.index,
                    confidence=0.92,
                    evidence="explicit preposition / functional marker",
                )

        # Bare noun routing: first suitable noun before the verb is Kartā;
        # first suitable noun after the verb is Karma. This is an English
        # prototype heuristic and intentionally lower confidence.
        nouns = [
            atom for atom in atom_set
            if atom.category in {"Noun", "NounOrModifier"}
            and atom.index != verb_atom.index
            and atom.index not in marker_role_by_index
        ]

        pre_verb = [atom for atom in nouns if atom.index < verb_atom.index]
        post_verb = [atom for atom in nouns if atom.index > verb_atom.index]

        if "Kartā" not in bindings and pre_verb:
            bindings["Kartā"] = Binding(
                role="Kartā",
                token=pre_verb[-1].token,
                token_index=pre_verb[-1].index,
                confidence=0.78,
                evidence="unmarked noun preceding action engine",
            )

        if "Karma" not in bindings and post_verb:
            # Prefer a noun close to the verb, while not taking a marker-bound
            # noun that already fills another role.
            candidate = post_verb[0]
            bindings["Karma"] = Binding(
                role="Karma",
                token=candidate.token,
                token_index=candidate.index,
                confidence=0.72,
                evidence="unmarked noun following action engine",
            )

        # Explicit marker-bound role should override a weak positional role.
        for role in ("Kartā", "Karma", "Karaṇa", "Sampradāna",
                     "Apādāna", "Adhikaraṇa", "Sambandha"):
            bindings.setdefault(
                role,
                Binding(
                    role=role,
                    token=None,
                    token_index=None,
                    confidence=0.0,
                    evidence="unresolved",
                ),
            )

        missing = [
            role for role in valency["required"]
            if bindings[role].token is None
        ]

        return {
            "Action_Engine": verb_atom.token,
            "Action_Engine_Index": verb_atom.index,
            "Valency": valency,
            "Bindings": {role: value.to_dict() for role, value in bindings.items()},
            "Missing_Required_Roles": missing,
            "Complete_Required_Frame": not missing,
        }

    # ------------------------------------------------------------------
    # Structural attention / generation constraints
    # ------------------------------------------------------------------

    def build_structural_bias(
        self,
        atom_set: Sequence[Atom],
        semantic_graph: Dict[str, Any],
        forward_bias: float = 5.0,
        reverse_scale: float = 0.5,
    ) -> List[List[float]]:
        n = len(atom_set)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        verb_index = semantic_graph["Action_Engine_Index"]

        for role, binding in semantic_graph["Bindings"].items():
            token_index = binding["token_index"]
            if token_index is None:
                continue

            matrix[token_index][verb_index] += forward_bias * binding["confidence"]
            matrix[verb_index][token_index] += (
                forward_bias * binding["confidence"] * reverse_scale
            )

        return matrix

    def simulate_asiddhatvam_mask(
        self,
        graph: Dict[str, Any],
        next_token_logits: Dict[str, float],
    ) -> Dict[str, float]:
        """Apply a transparent structural generation constraint.

        This is an experimental analogy to rule-boundary enforcement, not a
        claim that classical Asiddhatva directly defines neural logits.
        """
        masked = dict(next_token_logits)

        complete = graph["Complete_Required_Frame"]
        required = set(graph["Valency"]["required"])

        if complete:
            # Once required argument slots are filled, discourage generating
            # another known core argument noun. Modifiers/function words and
            # punctuation remain available.
            bound_nouns = {
                binding["token"]
                for role, binding in graph["Bindings"].items()
                if role in required and binding["token"]
            }
            for token in bound_nouns:
                if token in masked:
                    masked[token] = float("-inf")

        return masked

    # ------------------------------------------------------------------
    # Complete representation
    # ------------------------------------------------------------------

    def analyze(self, sentence: str) -> Dict[str, Any]:
        atoms = self.parse_to_atoms(sentence)
        graph = self.execute_valency_routing(atoms)
        graph["Sentence"] = sentence
        graph["Atoms"] = [atom.to_dict() for atom in atoms]
        graph["Structural_Bias"] = self.build_structural_bias(atoms, graph)
        graph["Architecture"] = {
            "surface": "English linear token stream",
            "symbolic_layer": "Vyakarana Graph Automata",
            "semantic_layer": "Paninian Kāraka / Ākāṅkṣā",
            "neural_layer": "Transformer-compatible additive bias",
            "generation_layer": "structural constraint gate",
        }
        return graph


# ---------------------------------------------------------------------------
# JSON / CLI
# ---------------------------------------------------------------------------

def save_json(graph: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(graph, handle, ensure_ascii=False, indent=2)


def render(graph: Dict[str, Any]) -> str:
    lines = [
        "=" * 72,
        "PANINIAN ENGLISH LLM — VYAKARAṆA GRAPH AUTOMATA",
        "=" * 72,
        f"Sentence: {graph['Sentence']}",
        f"Action Engine: {graph['Action_Engine']}",
        "",
        "VALENCY",
        f"  Required: {graph['Valency']['required']}",
        f"  Optional: {graph['Valency']['optional']}",
        "",
        "KĀRAKA BINDINGS",
    ]

    for role, binding in graph["Bindings"].items():
        if binding["token"]:
            lines.append(
                f"  {role:<12} -> {binding['token']:<16} "
                f"confidence={binding['confidence']:.2f}"
            )

    lines += [
        "",
        f"Missing required roles: {graph['Missing_Required_Roles'] or 'None'}",
        f"Complete frame: {graph['Complete_Required_Frame']}",
        "",
        "STRUCTURAL ATTENTION BIAS",
    ]

    for i, row in enumerate(graph["Structural_Bias"]):
        lines.append(
            f"  {i:02d}: " + " ".join(f"{v:6.2f}" for v in row)
        )

    return "\n".join(lines)


def self_test() -> None:
    engine = VyakaranaGraphAutomata()

    sentence = (
        "With a swift knife the merchant cut the ripe apple in the orchard."
    )
    graph = engine.analyze(sentence)

    assert graph["Action_Engine"] == "cut"
    assert graph["Bindings"]["Kartā"]["token"] == "merchant"
    assert graph["Bindings"]["Karma"]["token"] == "apple"
    assert graph["Bindings"]["Karaṇa"]["token"] == "knife"
    assert graph["Bindings"]["Adhikaraṇa"]["token"] == "orchard"
    assert graph["Complete_Required_Frame"] is True

    # The structural bias must connect each resolved role to the verb.
    verb = graph["Action_Engine_Index"]
    for role in ("Kartā", "Karma", "Karaṇa", "Adhikaraṇa"):
        idx = graph["Bindings"][role]["token_index"]
        assert graph["Structural_Bias"][idx][verb] > 0

    # Test recipient valency.
    give = engine.analyze("Rama gives a book to Sita.")
    assert give["Action_Engine"] == "gives"
    assert give["Bindings"]["Kartā"]["token"] == "rama"
    assert give["Bindings"]["Karma"]["token"] == "book"
    assert give["Bindings"]["Sampradāna"]["token"] == "sita"
    assert give["Complete_Required_Frame"] is True

    # Test generation mask.
    logits = {"merchant": 2.8, "apple": 1.5, "swiftly": 3.2}
    constrained = engine.simulate_asiddhatvam_mask(graph, logits)
    assert constrained["merchant"] == float("-inf")
    assert constrained["apple"] == float("-inf")
    assert constrained["swiftly"] == 3.2

    print("SELF_TEST_PASS")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paninian English Vyakarana Graph Automata prototype."
    )
    parser.add_argument("sentence", nargs="?", help="English sentence")
    parser.add_argument("--json", dest="json_path", help="Write JSON output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    sentence = args.sentence or (
        "With a swift knife the merchant cut the ripe apple in the orchard."
    )
    graph = VyakaranaGraphAutomata().analyze(sentence)
    print(render(graph))

    if args.json_path:
        save_json(graph, args.json_path)
        print(f"\nJSON written to: {args.json_path}")


if __name__ == "__main__":
    main()
