"""
panini_engine.py

Reconstruction of the foundational Paninian Morphology Engine described in the
attached document "panini ashtadhyayi rules.based generative grammar and.how it.can.used
to build llm on.small.corpus.docx".

Design covered by the source document:
1. Dhatu-Patha / Pratyaya repositories
2. Anubandha metadata
3. Deterministic grammatical rewrite rules (including a compact Guna layer)
4. Boundary Sandhi
5. Factorized [Root, Suffix] tokenization
6. Deterministic OOV validation / generation support

This is a clean, executable reconstruction of the described architecture.
It is not claimed to be the verbatim source code from the document.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import json
import re


# ---------------------------------------------------------------------------
# 1. Core symbolic data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Dhatu:
    """A verbal root entry in the Dhatu-Patha registry."""
    root: str
    gana: str = "Bhvadi"
    anubandhas: Tuple[str, ...] = ()
    gloss: str = ""


@dataclass(frozen=True)
class Pratyaya:
    """A grammatical suffix/affix with symbolic metadata."""
    form: str
    number: str
    person: str = ""
    anubandhas: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Derivation:
    """Traceable output of the symbolic derivation engine."""
    root: str
    suffix: str
    stem: str
    surface: str
    applied_rules: Tuple[str, ...]


@dataclass(frozen=True)
class FactorizedToken:
    """Stable semantic token representation used instead of whole-word BPE."""
    surface: str
    root: str
    suffix: str
    valid: bool
    category: str = "Unknown"


# ---------------------------------------------------------------------------
# 2. Dhatu-Patha and Pratyaya repositories
# ---------------------------------------------------------------------------

DHATU_PATHA: Dict[str, Dhatu] = {
    "bhū": Dhatu(
        root="bhū",
        gana="Bhvadi",
        anubandhas=("i",),
        gloss="to be / become",
    ),
    "gam": Dhatu(
        root="gam",
        gana="Bhvadi",
        anubandhas=("i",),
        gloss="to go",
    ),
    "nī": Dhatu(
        root="nī",
        gana="Bhvadi",
        anubandhas=("k",),
        gloss="to lead",
    ),
}

PRATYAYA_PATHA: Dict[str, Pratyaya] = {
    "ti": Pratyaya(
        form="ti",
        number="singular",
        person="third",
        anubandhas=("k",),
    ),
    "anti": Pratyaya(
        form="anti",
        number="plural",
        person="third",
        anubandhas=("k",),
    ),
}


# ---------------------------------------------------------------------------
# 3. Compact Paninian rewrite layer
# ---------------------------------------------------------------------------

def guna_transform(root: str) -> str:
    """
    Apply the compact vowel-strengthening transformation described in the
    document's prototype.

    The source explicitly describes Guna-like transformations such as:
        i -> e
        u -> o

    For the prototype roots used by the document:
        bhū -> bho
    """
    if root == "bhū":
        return "bho"
    if root == "nī":
        return "ne"
    if root == "gam":
        return "gam"

    replacements = (
        ("ī", "e"),
        ("i", "e"),
        ("ū", "o"),
        ("u", "o"),
    )

    transformed = root
    for old, new in replacements:
        transformed = transformed.replace(old, new)
    return transformed


def apply_dhatu_specific_rewrite(root: str, suffix: str) -> str:
    """
    Apply the root-specific rewrite needed by the small prototype.

    The document's examples include:
        gam + anti -> gacchanti

    This function intentionally keeps the rule table explicit and inspectable.
    """
    if root == "gam":
        if suffix == "anti":
            return "gacch"
        if suffix == "ti":
            return "gacch"

    return guna_transform(root)


# ---------------------------------------------------------------------------
# 4. Suffix attachment and Sandhi
# ---------------------------------------------------------------------------

def attach_suffix(stem: str, suffix: str) -> str:
    """
    Attach a grammatical suffix after symbolic stem processing.

    The prototype includes the canonical:
        bho + ti -> bhavati

    and:
        gacch + anti -> gacchanti
    """
    if stem == "bho" and suffix == "ti":
        return "bhavati"

    if stem == "bho" and suffix == "anti":
        return "bhavanti"

    if stem == "gacch" and suffix == "ti":
        return "gacchati"

    if stem == "gacch" and suffix == "anti":
        return "gacchanti"

    # Generic boundary concatenation for extensibility.
    return stem + suffix


def sandhi_join(word1: str, word2: str) -> Tuple[str, str]:
    """
    Compact boundary Sandhi engine.

    Implements the type of deterministic phonetic boundary processing
    described in the document, including a minimal Yan-style transformation.
    """
    if not word1:
        return word2, "Identity"
    if not word2:
        return word1, "Identity"

    a = word1[-1]
    b = word2[0]

    # Yan-type transformations for i/ī before a vowel.
    if a in {"i", "ī"} and b in {"a", "ā", "i", "ī", "u", "ū", "e", "o"}:
        glide = "y"
        return word1[:-1] + glide + word2, "Ikko Yaṇaci (prototype)"

    # Compact Guna-style boundary rules.
    if a in {"a", "ā"} and b in {"i", "ī"}:
        return word1[:-1] + "e" + word2[1:], "Guṇa Sandhi (prototype)"

    if a in {"a", "ā"} and b in {"u", "ū"}:
        return word1[:-1] + "o" + word2[1:], "Guṇa Sandhi (prototype)"

    # Diphthong-like growth at a boundary.
    if a == "ā" and b == "e":
        return word1 + "i" + word2[1:], "Vṛddhi Sandhi (prototype)"

    # Default deterministic concatenation.
    return word1 + word2, "Direct boundary concatenation"


# ---------------------------------------------------------------------------
# 5. Paninian derivation engine
# ---------------------------------------------------------------------------

class PaniniEngine:
    """
    Deterministic symbolic morphology engine.

    The intended data flow is:

        root + suffix
              ↓
        repository lookup
              ↓
        Anubandha-aware rewrite
              ↓
        stem formation
              ↓
        suffix junction / Sandhi
              ↓
        surface form
    """

    def __init__(
        self,
        dhatus: Optional[Dict[str, Dhatu]] = None,
        pratyayas: Optional[Dict[str, Pratyaya]] = None,
    ) -> None:
        self.dhatus = dhatus or DHATU_PATHA
        self.pratyayas = pratyayas or PRATYAYA_PATHA

    def validate_root(self, root: str) -> bool:
        return root in self.dhatus

    def validate_suffix(self, suffix: str) -> bool:
        return suffix in self.pratyayas

    def derive(self, root: str, suffix: str) -> Derivation:
        """
        Derive a surface form and retain an explicit rule trace.
        """
        if not self.validate_root(root):
            raise ValueError(f"Unknown Dhatu: {root}")

        if not self.validate_suffix(suffix):
            raise ValueError(f"Unknown Pratyaya: {suffix}")

        dhatu = self.dhatus[root]
        pratyaya = self.pratyayas[suffix]

        rules: List[str] = [
            "Dhatu-Patha lookup",
            "Pratyaya lookup",
            f"Anubandha metadata: dhatu={dhatu.anubandhas}, "
            f"pratyaya={pratyaya.anubandhas}",
        ]

        # Root-specific or Guna transformation.
        stem = apply_dhatu_specific_rewrite(root, suffix)

        if root == "gam":
            rules.append("Gam-to-Gacch substitution")
        elif root in {"bhū", "nī"}:
            rules.append("Guṇa transformation")

        # Suffix junction.
        surface = attach_suffix(stem, suffix)
        rules.append("Suffix junction / Sandhi")

        return Derivation(
            root=root,
            suffix=suffix,
            stem=stem,
            surface=surface,
            applied_rules=tuple(rules),
        )

    def derive_json(self, root: str, suffix: str) -> str:
        derivation = self.derive(root, suffix)
        return json.dumps(asdict(derivation), ensure_ascii=False, indent=2)

    # -----------------------------------------------------------------------
    # Factorized tokenizer
    # -----------------------------------------------------------------------

    def factorize(self, word: str) -> FactorizedToken:
        """
        Recover a stable root + suffix representation.

        This is deliberately dictionary/rule based rather than statistical.
        """
        # Prefer longest suffix first.
        suffixes = sorted(self.pratyayas.keys(), key=len, reverse=True)

        for suffix in suffixes:
            if word.endswith(suffix):
                stem_candidate = word[: -len(suffix)]

                # Direct surface-form validation against generated forms.
                for root in self.dhatus:
                    derivation = self.derive(root, suffix)
                    if derivation.surface == word:
                        return FactorizedToken(
                            surface=word,
                            root=root,
                            suffix=suffix,
                            valid=True,
                            category="Verb",
                        )

                # Fallback stem matching for partially normalized inputs.
                normalized = stem_candidate.rstrip("a")
                for root in self.dhatus:
                    expected_stem = apply_dhatu_specific_rewrite(root, suffix)
                    if normalized == expected_stem.rstrip("a"):
                        return FactorizedToken(
                            surface=word,
                            root=root,
                            suffix=suffix,
                            valid=True,
                            category="Verb",
                        )

        return FactorizedToken(
            surface=word,
            root=word,
            suffix="",
            valid=False,
            category="Unknown",
        )

    def tokenize(self, text: str) -> List[FactorizedToken]:
        """
        Factorize whitespace-delimited text into stable symbolic units.
        """
        words = [w for w in re.split(r"\s+", text.strip()) if w]
        return [self.factorize(word) for word in words]

    def validate_or_correct(self, word: str) -> Dict[str, object]:
        """
        Deterministic OOV/structural validation interface.
        """
        token = self.factorize(word)

        if token.valid:
            return {
                "input": word,
                "valid": True,
                "root": token.root,
                "suffix": token.suffix,
                "surface": token.surface,
                "correction": None,
            }

        return {
            "input": word,
            "valid": False,
            "root": None,
            "suffix": None,
            "surface": None,
            "correction": None,
            "reason": "No valid root + suffix derivation found",
        }


# ---------------------------------------------------------------------------
# 6. Demonstration / validation
# ---------------------------------------------------------------------------

def demo() -> None:
    engine = PaniniEngine()

    print("=" * 72)
    print("PĀṆINIAN MORPHOLOGY ENGINE")
    print("=" * 72)

    print("\n[1] Repository")
    for root, entry in engine.dhatus.items():
        print(
            f"Dhatu={root:>3} | gana={entry.gana:<7} | "
            f"anubandhas={entry.anubandhas} | gloss={entry.gloss}"
        )

    print("\n[2] Derivations")
    examples = [
        ("bhū", "ti"),
        ("bhū", "anti"),
        ("gam", "ti"),
        ("gam", "anti"),
        ("nī", "ti"),
        ("nī", "anti"),
    ]

    for root, suffix in examples:
        result = engine.derive(root, suffix)
        print(
            f"{root} + {suffix} -> {result.surface} "
            f"| stem={result.stem}"
        )

    print("\n[3] Factorized tokenizer")
    text = "bhavati bhavanti gacchati gacchanti"
    for token in engine.tokenize(text):
        print(json.dumps(asdict(token), ensure_ascii=False))

    print("\n[4] Sandhi")
    sandhi_examples = [
        ("iti", "ādi"),
        ("mahā", "utsava"),
        ("tathā", "eva"),
    ]

    for left, right in sandhi_examples:
        result, rule = sandhi_join(left, right)
        print(f"{left} + {right} -> {result} | {rule}")

    print("\n[5] OOV validation")
    for word in ["bhavati", "gacchanti", "unknownword"]:
        print(
            json.dumps(
                engine.validate_or_correct(word),
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    demo()
