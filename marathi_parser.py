"""
marathi_parser.py
=================

Panini Language Machine — Marathi Morphological & Grammatical Parser

File 5/14.

Purpose
-------
Provide the first language-facing parser on top of the Paninian compiler
stack.

Pipeline:

    Marathi sentence
        ↓
    Unicode normalization
        ↓
    sentence segmentation
        ↓
    tokenization
        ↓
    clitic / punctuation handling
        ↓
    lexical lookup
        ↓
    morphological feature extraction
        ↓
    Paninian grammatical feature representation
        ↓
    parser output
        ↓
    future Karaka/dependency layer

Scope
-----
This file is intentionally a deterministic parser framework rather than a
claim of complete Marathi morphology. A full Marathi lexicon, inflectional
paradigm database, sandhi rules, and validated Paninian rule corpus can be
plugged into the interfaces defined here.

The parser is designed around the principle:

    surface form
        → candidate analyses
        → grammatical feature structure
        → symbolic rule engine

This keeps linguistic structure explicit and inspectable.

The implementation supports:
    * Devanagari normalization
    * robust tokenization
    * punctuation recognition
    * lexical entries
    * suffix-pattern analysis
    * gender / number / person / case / tense / aspect / mood features
    * POS hypotheses
    * confidence scoring
    * ambiguity preservation
    * parser traces
    * JSON serialization
    * batch parsing
    * deterministic prototype lexicon

It does not silently invent a complete Marathi grammar.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import re
import unicodedata


# ============================================================================
# 1. Enumerations
# ============================================================================

class TokenKind(str, Enum):
    WORD = "word"
    NUMBER = "number"
    PUNCTUATION = "punctuation"
    SYMBOL = "symbol"
    WHITESPACE = "whitespace"


class PartOfSpeech(str, Enum):
    NOUN = "noun"
    PRONOUN = "pronoun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    POSTPOSITION = "postposition"
    DETERMINER = "determiner"
    CONJUNCTION = "conjunction"
    PARTICLE = "particle"
    INTERJECTION = "interjection"
    AUXILIARY = "auxiliary"
    UNKNOWN = "unknown"


class Gender(str, Enum):
    MASCULINE = "masculine"
    FEMININE = "feminine"
    NEUTER = "neuter"
    UNKNOWN = "unknown"


class Number(str, Enum):
    SINGULAR = "singular"
    PLURAL = "plural"
    UNKNOWN = "unknown"


class Person(str, Enum):
    FIRST = "first"
    SECOND = "second"
    THIRD = "third"
    UNKNOWN = "unknown"


class Case(str, Enum):
    NOMINATIVE = "nominative"
    ACCUSATIVE = "accusative"
    INSTRUMENTAL = "instrumental"
    DATIVE = "dative"
    ABLATIVE = "ablative"
    GENITIVE = "genitive"
    LOCATIVE = "locative"
    VOCATIVE = "vocative"
    UNKNOWN = "unknown"


class Tense(str, Enum):
    PRESENT = "present"
    PAST = "past"
    FUTURE = "future"
    UNKNOWN = "unknown"


class Aspect(str, Enum):
    PERFECTIVE = "perfective"
    IMPERFECTIVE = "imperfective"
    PROGRESSIVE = "progressive"
    HABITUAL = "habitual"
    UNKNOWN = "unknown"


class Mood(str, Enum):
    INDICATIVE = "indicative"
    IMPERATIVE = "imperative"
    SUBJUNCTIVE = "subjunctive"
    CONDITIONAL = "conditional"
    UNKNOWN = "unknown"


# ============================================================================
# 2. Normalization
# ============================================================================

class MarathiNormalizer:
    """
    Unicode-safe normalizer.

    Marathi is represented primarily in Devanagari. The normalizer uses
    Unicode NFC and removes zero-width formatting characters that otherwise
    create hard-to-debug token mismatches.

    It does not transliterate text.
    """

    ZERO_WIDTH = {
        "\u200b",  # zero width space
        "\u200c",  # zero width non-joiner
        "\u200d",  # zero width joiner
        "\ufeff",  # BOM
    }

    @classmethod
    def normalize(cls, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        text = unicodedata.normalize("NFC", text)

        for char in cls.ZERO_WIDTH:
            text = text.replace(char, "")

        # Normalize repeated whitespace while preserving line boundaries.
        text = re.sub(r"[ \t\r\f\v]+", " ", text)

        return text.strip()


# ============================================================================
# 3. Tokenization
# ============================================================================

@dataclass(frozen=True)
class Token:
    index: int
    text: str
    kind: TokenKind
    start: int
    end: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "kind": self.kind.value,
            "start": self.start,
            "end": self.end,
        }


class MarathiTokenizer:
    """
    Tokenizer for Devanagari + common Indian/Unicode punctuation.

    The tokenizer intentionally preserves punctuation as tokens because
    downstream sentence/dependency parsing can use punctuation boundaries.
    """

    WORD_PATTERN = re.compile(
        r"[\u0900-\u097F]+"
    )

    NUMBER_PATTERN = re.compile(
        r"(?:\d+(?:[.,]\d+)?)"
    )

    PUNCTUATION = set(
        ".,!?;:।॥,!?;:()[]{}\"'“”‘’«»—-–"
    )

    @classmethod
    def _kind(cls, text: str) -> TokenKind:
        if cls.NUMBER_PATTERN.fullmatch(text):
            return TokenKind.NUMBER

        if len(text) == 1 and text in cls.PUNCTUATION:
            return TokenKind.PUNCTUATION

        if cls.WORD_PATTERN.fullmatch(text):
            return TokenKind.WORD

        return TokenKind.SYMBOL

    @classmethod
    def tokenize(cls, text: str) -> List[Token]:
        text = MarathiNormalizer.normalize(text)

        tokens: List[Token] = []

        pattern = re.compile(
            r"[\u0900-\u097F]+"
            r"|(?:\d+(?:[.,]\d+)?)"
            r"|[.,!?;:।॥()\[\]{}\"'“”‘’«»—–-]"
            r"|[^\s]"
        )

        for index, match in enumerate(pattern.finditer(text)):
            value = match.group(0)

            tokens.append(
                Token(
                    index=index,
                    text=value,
                    kind=cls._kind(value),
                    start=match.start(),
                    end=match.end(),
                )
            )

        return tokens


# ============================================================================
# 4. Grammatical feature bundle
# ============================================================================

@dataclass
class MorphFeatures:
    """
    Explicit feature structure used by later Paninian/Karaka layers.

    Values remain strings/enums rather than opaque embeddings so that every
    decision can be inspected and transformed into a symbolic representation.
    """

    pos: PartOfSpeech = PartOfSpeech.UNKNOWN
    lemma: Optional[str] = None
    gender: Gender = Gender.UNKNOWN
    number: Number = Number.UNKNOWN
    person: Person = Person.UNKNOWN
    case: Case = Case.UNKNOWN
    tense: Tense = Tense.UNKNOWN
    aspect: Aspect = Aspect.UNKNOWN
    mood: Mood = Mood.UNKNOWN

    # Marathi-specific/grammar-neutral extension fields.
    definiteness: Optional[str] = None
    animacy: Optional[str] = None
    transitivity: Optional[str] = None
    evidentiality: Optional[str] = None

    # Paninian preparation fields.
    karaka_candidate: Optional[str] = None
    vibhakti_marker: Optional[str] = None
    dhatu: Optional[str] = None
    pratyaya: Optional[str] = None
    lakara: Optional[str] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)

        for key, value in list(result.items()):
            if isinstance(value, Enum):
                result[key] = value.value

        return result


# ============================================================================
# 5. Lexical entries
# ============================================================================

@dataclass(frozen=True)
class LexicalEntry:
    surface: str
    lemma: str
    pos: PartOfSpeech

    gender: Gender = Gender.UNKNOWN
    number: Number = Number.UNKNOWN
    person: Person = Person.UNKNOWN
    case: Case = Case.UNKNOWN

    tense: Tense = Tense.UNKNOWN
    aspect: Aspect = Aspect.UNKNOWN
    mood: Mood = Mood.UNKNOWN

    dhatu: Optional[str] = None
    pratyaya: Optional[str] = None
    vibhakti_marker: Optional[str] = None
    lakara: Optional[str] = None

    features: Mapping[str, Any] = field(default_factory=dict)

    def to_features(self) -> MorphFeatures:
        return MorphFeatures(
            pos=self.pos,
            lemma=self.lemma,
            gender=self.gender,
            number=self.number,
            person=self.person,
            case=self.case,
            tense=self.tense,
            aspect=self.aspect,
            mood=self.mood,
            dhatu=self.dhatu,
            pratyaya=self.pratyaya,
            vibhakti_marker=self.vibhakti_marker,
            lakara=self.lakara,
            extra=dict(self.features),
        )


class MarathiLexicon:
    """
    Deterministic lexical store.

    Multiple entries for the same surface form are allowed. This is essential
    because Marathi forms can be morphologically/grammatically ambiguous.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, List[LexicalEntry]] = {}

    def add(self, entry: LexicalEntry) -> None:
        self._entries.setdefault(entry.surface, []).append(entry)

    def add_many(self, entries: Iterable[LexicalEntry]) -> None:
        for entry in entries:
            self.add(entry)

    def lookup(self, surface: str) -> List[LexicalEntry]:
        return list(self._entries.get(surface, []))

    def __len__(self) -> int:
        return sum(len(items) for items in self._entries.values())

    def surfaces(self) -> List[str]:
        return sorted(self._entries)


# ============================================================================
# 6. Morphological suffix rules
# ============================================================================

@dataclass(frozen=True)
class SuffixRule:
    suffix: str
    pos: PartOfSpeech
    score: float
    feature_updates: Mapping[str, Any] = field(default_factory=dict)
    lemma_strategy: str = "remove_suffix"
    label: str = ""

    def apply(self, surface: str) -> Optional[Tuple[str, MorphFeatures]]:
        if not surface.endswith(self.suffix):
            return None

        if self.suffix:
            lemma = surface[: -len(self.suffix)]
        else:
            lemma = surface

        if not lemma:
            return None

        features = MorphFeatures(
            pos=self.pos,
            lemma=lemma,
            extra={
                "analysis_method": "suffix_rule",
                "suffix_rule": self.label or self.suffix,
            },
        )

        for key, value in self.feature_updates.items():
            if hasattr(features, key):
                setattr(features, key, value)
            else:
                features.extra[key] = value

        return lemma, features


# ============================================================================
# 7. Candidate analyses
# ============================================================================

@dataclass
class MorphAnalysis:
    token: Token
    features: MorphFeatures
    score: float
    source: str
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token.to_dict(),
            "features": self.features.to_dict(),
            "score": self.score,
            "source": self.source,
            "evidence": list(self.evidence),
        }


# ============================================================================
# 8. Parser trace
# ============================================================================

@dataclass
class ParseTrace:
    stage: str
    message: str
    token_index: Optional[int] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "message": self.message,
            "token_index": self.token_index,
            "data": self.data,
        }


# ============================================================================
# 9. Sentence parse
# ============================================================================

@dataclass
class ParsedSentence:
    text: str
    tokens: List[Token]
    analyses: Dict[int, List[MorphAnalysis]]
    traces: List[ParseTrace] = field(default_factory=list)

    def best_analysis(self, token_index: int) -> Optional[MorphAnalysis]:
        candidates = self.analyses.get(token_index, [])

        if not candidates:
            return None

        return max(candidates, key=lambda item: item.score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "tokens": [token.to_dict() for token in self.tokens],
            "analyses": {
                str(index): [
                    analysis.to_dict()
                    for analysis in candidates
                ]
                for index, candidates in self.analyses.items()
            },
            "traces": [
                trace.to_dict()
                for trace in self.traces
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
        )


# ============================================================================
# 10. Marathi parser
# ============================================================================

class MarathiParser:
    """
    Deterministic Marathi morphological parser.

    Parsing order:

        1. normalize
        2. tokenize
        3. punctuation/number classification
        4. exact lexical lookup
        5. suffix-rule analysis
        6. unknown-token fallback
        7. score and preserve ambiguity
    """

    def __init__(
        self,
        lexicon: Optional[MarathiLexicon] = None,
        suffix_rules: Optional[Sequence[SuffixRule]] = None,
    ) -> None:
        self.lexicon = lexicon or build_prototype_marathi_lexicon()

        self.suffix_rules = list(
            suffix_rules or build_prototype_suffix_rules()
        )

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _punctuation_analysis(token: Token) -> MorphAnalysis:
        features = MorphFeatures(
            pos=PartOfSpeech.PARTICLE,
            lemma=token.text,
            extra={"punctuation": True},
        )

        return MorphAnalysis(
            token=token,
            features=features,
            score=1.0,
            source="tokenizer",
            evidence=["punctuation classification"],
        )

    @staticmethod
    def _number_analysis(token: Token) -> MorphAnalysis:
        features = MorphFeatures(
            pos=PartOfSpeech.DETERMINER,
            lemma=token.text,
            number=Number.UNKNOWN,
            extra={"numeric": True},
        )

        return MorphAnalysis(
            token=token,
            features=features,
            score=0.9,
            source="tokenizer",
            evidence=["numeric token"],
        )

    def _lexical_analyses(
        self,
        token: Token,
    ) -> List[MorphAnalysis]:
        entries = self.lexicon.lookup(token.text)

        analyses: List[MorphAnalysis] = []

        for entry in entries:
            analyses.append(
                MorphAnalysis(
                    token=token,
                    features=entry.to_features(),
                    score=1.0,
                    source="lexicon",
                    evidence=[
                        f"exact lexical match: {entry.lemma}",
                        f"POS={entry.pos.value}",
                    ],
                )
            )

        return analyses

    def _suffix_analyses(
        self,
        token: Token,
    ) -> List[MorphAnalysis]:
        analyses: List[MorphAnalysis] = []

        for rule in self.suffix_rules:
            result = rule.apply(token.text)

            if result is None:
                continue

            lemma, features = result

            analyses.append(
                MorphAnalysis(
                    token=token,
                    features=features,
                    score=rule.score,
                    source="suffix_rule",
                    evidence=[
                        f"suffix={rule.suffix}",
                        rule.label or "suffix pattern",
                        f"candidate lemma={lemma}",
                    ],
                )
            )

        return analyses

    @staticmethod
    def _unknown_analysis(
        token: Token,
    ) -> MorphAnalysis:
        features = MorphFeatures(
            pos=PartOfSpeech.UNKNOWN,
            lemma=token.text,
            extra={
                "analysis_method": "unknown",
                "requires_lexicon_or_rule": True,
            },
        )

        return MorphAnalysis(
            token=token,
            features=features,
            score=0.05,
            source="fallback",
            evidence=["no lexical or suffix-rule analysis"],
        )

    @staticmethod
    def _deduplicate(
        analyses: Sequence[MorphAnalysis],
    ) -> List[MorphAnalysis]:
        seen: set[Tuple[Any, ...]] = set()
        result: List[MorphAnalysis] = []

        for analysis in analyses:
            features = analysis.features

            key = (
                features.lemma,
                features.pos.value,
                features.gender.value,
                features.number.value,
                features.case.value,
                features.tense.value,
                features.aspect.value,
                features.mood.value,
                analysis.source,
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(analysis)

        result.sort(
            key=lambda item: (
                -item.score,
                item.source,
                item.features.pos.value,
            )
        )

        return result

    # ------------------------------------------------------------------
    # Public parse
    # ------------------------------------------------------------------

    def parse(self, text: str) -> ParsedSentence:
        normalized = MarathiNormalizer.normalize(text)
        tokens = MarathiTokenizer.tokenize(normalized)

        traces: List[ParseTrace] = [
            ParseTrace(
                stage="normalize",
                message="Unicode-normalized Marathi input",
                data={"normalized": normalized},
            ),
            ParseTrace(
                stage="tokenize",
                message="Tokenized sentence",
                data={"token_count": len(tokens)},
            ),
        ]

        analyses: Dict[int, List[MorphAnalysis]] = {}

        for token in tokens:
            if token.kind == TokenKind.PUNCTUATION:
                candidates = [
                    self._punctuation_analysis(token)
                ]

            elif token.kind == TokenKind.NUMBER:
                candidates = [
                    self._number_analysis(token)
                ]

            elif token.kind == TokenKind.WORD:
                lexical = self._lexical_analyses(token)
                suffix = self._suffix_analyses(token)

                candidates = lexical + suffix

                if not candidates:
                    candidates = [
                        self._unknown_analysis(token)
                    ]

            else:
                candidates = [
                    self._unknown_analysis(token)
                ]

            candidates = self._deduplicate(candidates)
            analyses[token.index] = candidates

            traces.append(
                ParseTrace(
                    stage="morphology",
                    message="Generated morphological candidates",
                    token_index=token.index,
                    data={
                        "surface": token.text,
                        "candidate_count": len(candidates),
                    },
                )
            )

        return ParsedSentence(
            text=normalized,
            tokens=tokens,
            analyses=analyses,
            traces=traces,
        )

    def parse_many(
        self,
        texts: Iterable[str],
    ) -> List[ParsedSentence]:
        return [self.parse(text) for text in texts]


# ============================================================================
# 11. Prototype Marathi lexicon
# ============================================================================

def build_prototype_marathi_lexicon() -> MarathiLexicon:
    """
    Small deterministic vocabulary for integration testing.

    These entries are examples for exercising the parser architecture.
    They are not intended to constitute a complete Marathi lexicon.
    """
    lexicon = MarathiLexicon()

    lexicon.add_many(
        [
            LexicalEntry(
                surface="राम",
                lemma="राम",
                pos=PartOfSpeech.NOUN,
                gender=Gender.MASCULINE,
                number=Number.SINGULAR,
                case=Case.NOMINATIVE,
            ),
            LexicalEntry(
                surface="सीता",
                lemma="सीता",
                pos=PartOfSpeech.NOUN,
                gender=Gender.FEMININE,
                number=Number.SINGULAR,
                case=Case.NOMINATIVE,
            ),
            LexicalEntry(
                surface="घर",
                lemma="घर",
                pos=PartOfSpeech.NOUN,
                gender=Gender.NEUTER,
                number=Number.SINGULAR,
                case=Case.NOMINATIVE,
            ),
            LexicalEntry(
                surface="पुस्तक",
                lemma="पुस्तक",
                pos=PartOfSpeech.NOUN,
                gender=Gender.NEUTER,
                number=Number.SINGULAR,
                case=Case.NOMINATIVE,
            ),
            LexicalEntry(
                surface="मी",
                lemma="मी",
                pos=PartOfSpeech.PRONOUN,
                number=Number.SINGULAR,
                person=Person.FIRST,
                case=Case.NOMINATIVE,
            ),
            LexicalEntry(
                surface="तू",
                lemma="तू",
                pos=PartOfSpeech.PRONOUN,
                number=Number.SINGULAR,
                person=Person.SECOND,
                case=Case.NOMINATIVE,
            ),
            LexicalEntry(
                surface="तो",
                lemma="तो",
                pos=PartOfSpeech.PRONOUN,
                gender=Gender.MASCULINE,
                number=Number.SINGULAR,
                person=Person.THIRD,
                case=Case.NOMINATIVE,
            ),
            LexicalEntry(
                surface="ती",
                lemma="ती",
                pos=PartOfSpeech.PRONOUN,
                gender=Gender.FEMININE,
                number=Number.SINGULAR,
                person=Person.THIRD,
                case=Case.NOMINATIVE,
            ),
            LexicalEntry(
                surface="आहे",
                lemma="अस",
                pos=PartOfSpeech.VERB,
                tense=Tense.PRESENT,
                mood=Mood.INDICATIVE,
                dhatu="अस्",
                lakara="लट्",
            ),
            LexicalEntry(
                surface="होता",
                lemma="हो",
                pos=PartOfSpeech.VERB,
                gender=Gender.MASCULINE,
                number=Number.SINGULAR,
                tense=Tense.PAST,
                aspect=Aspect.IMPERFECTIVE,
                mood=Mood.INDICATIVE,
                dhatu="हो",
            ),
            LexicalEntry(
                surface="गेला",
                lemma="जा",
                pos=PartOfSpeech.VERB,
                gender=Gender.MASCULINE,
                number=Number.SINGULAR,
                tense=Tense.PAST,
                aspect=Aspect.PERFECTIVE,
                mood=Mood.INDICATIVE,
                dhatu="गम्",
            ),
            LexicalEntry(
                surface="गेली",
                lemma="जा",
                pos=PartOfSpeech.VERB,
                gender=Gender.FEMININE,
                number=Number.SINGULAR,
                tense=Tense.PAST,
                aspect=Aspect.PERFECTIVE,
                mood=Mood.INDICATIVE,
                dhatu="गम्",
            ),
            LexicalEntry(
                surface="आणि",
                lemma="आणि",
                pos=PartOfSpeech.CONJUNCTION,
            ),
            LexicalEntry(
                surface="पण",
                lemma="पण",
                pos=PartOfSpeech.CONJUNCTION,
            ),
            LexicalEntry(
                surface="मध्ये",
                lemma="मध्ये",
                pos=PartOfSpeech.POSTPOSITION,
                case=Case.LOCATIVE,
                vibhakti_marker="मध्ये",
            ),
            LexicalEntry(
                surface="ला",
                lemma="ला",
                pos=PartOfSpeech.POSTPOSITION,
                case=Case.DATIVE,
                vibhakti_marker="ला",
            ),
            LexicalEntry(
                surface="ने",
                lemma="ने",
                pos=PartOfSpeech.POSTPOSITION,
                case=Case.INSTRUMENTAL,
                vibhakti_marker="ने",
            ),
        ]
    )

    return lexicon


# ============================================================================
# 12. Prototype suffix rules
# ============================================================================

def build_prototype_suffix_rules() -> List[SuffixRule]:
    """
    Conservative suffix patterns for exercising candidate generation.

    They intentionally return hypotheses rather than pretending that suffix
    stripping alone proves a complete morphological analysis.
    """
    return [
        SuffixRule(
            suffix="ांनी",
            pos=PartOfSpeech.NOUN,
            score=0.72,
            feature_updates={
                "number": Number.PLURAL,
                "case": Case.INSTRUMENTAL,
                "vibhakti_marker": "नी",
            },
            label="plural-instrumental-pattern",
        ),
        SuffixRule(
            suffix="ना",
            pos=PartOfSpeech.VERB,
            score=0.62,
            feature_updates={
                "extra": {
                    "nonfinite": True,
                }
            },
            label="verb-infinitive-pattern",
        ),
        SuffixRule(
            suffix="ले",
            pos=PartOfSpeech.VERB,
            score=0.55,
            feature_updates={
                "tense": Tense.PAST,
            },
            label="past-pattern",
        ),
        SuffixRule(
            suffix="ला",
            pos=PartOfSpeech.POSTPOSITION,
            score=0.50,
            feature_updates={
                "case": Case.DATIVE,
                "vibhakti_marker": "ला",
            },
            label="dative-pattern",
        ),
        SuffixRule(
            suffix="मध्ये",
            pos=PartOfSpeech.POSTPOSITION,
            score=0.85,
            feature_updates={
                "case": Case.LOCATIVE,
                "vibhakti_marker": "मध्ये",
            },
            label="locative-pattern",
        ),
    ]


# ============================================================================
# 13. Paninian feature bridge
# ============================================================================

@dataclass
class PaninianToken:
    """
    Intermediate representation passed to the later Paninian/Karaka layers.
    """

    token_index: int
    surface: str
    lemma: Optional[str]
    pos: str
    features: Dict[str, Any]
    candidate_score: float
    analysis_source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PaninianFeatureBridge:
    """
    Convert parser output into a stable symbolic feature representation.

    This bridge deliberately does not assign final Karaka relations. That is
    File 8's responsibility.
    """

    @staticmethod
    def best_tokens(
        parsed: ParsedSentence,
    ) -> List[PaninianToken]:
        result: List[PaninianToken] = []

        for token in parsed.tokens:
            analysis = parsed.best_analysis(token.index)

            if analysis is None:
                continue

            features = analysis.features.to_dict()

            result.append(
                PaninianToken(
                    token_index=token.index,
                    surface=token.text,
                    lemma=analysis.features.lemma,
                    pos=analysis.features.pos.value,
                    features=features,
                    candidate_score=analysis.score,
                    analysis_source=analysis.source,
                )
            )

        return result

    @staticmethod
    def to_context(
        parsed: ParsedSentence,
    ) -> Dict[str, Any]:
        tokens = PaninianFeatureBridge.best_tokens(parsed)

        return {
            "language": "mr",
            "script": "Devanagari",
            "sentence": parsed.text,
            "tokens": [token.to_dict() for token in tokens],
            "token_count": len(tokens),
            "representation": "paninian_feature_bundle",
        }


# ============================================================================
# 14. Convenience API
# ============================================================================

_DEFAULT_PARSER: Optional[MarathiParser] = None


def get_default_parser() -> MarathiParser:
    global _DEFAULT_PARSER

    if _DEFAULT_PARSER is None:
        _DEFAULT_PARSER = MarathiParser()

    return _DEFAULT_PARSER


def parse_marathi(text: str) -> ParsedSentence:
    return get_default_parser().parse(text)


def parse_to_paninian_context(text: str) -> Dict[str, Any]:
    parsed = parse_marathi(text)
    return PaninianFeatureBridge.to_context(parsed)


# ============================================================================
# 15. Demonstration
# ============================================================================

def demo() -> None:
    print("=" * 80)
    print("PANINI LANGUAGE MACHINE — MARATHI PARSER")
    print("=" * 80)

    sentence = "राम पुस्तक घरी घेऊन गेला."

    parser = MarathiParser()
    parsed = parser.parse(sentence)

    print("\nInput:")
    print(f"  {sentence}")

    print("\nTokens:")

    for token in parsed.tokens:
        print(
            f"  [{token.index}] {token.text:<12} "
            f"{token.kind.value}"
        )

    print("\nBest morphological analyses:")

    for token in parsed.tokens:
        analysis = parsed.best_analysis(token.index)

        if analysis is None:
            continue

        features = analysis.features

        print(
            f"  [{token.index}] {token.text:<12} "
            f"POS={features.pos.value:<12} "
            f"lemma={str(features.lemma):<10} "
            f"source={analysis.source:<12} "
            f"score={analysis.score:.2f}"
        )

    print("\nPaninian feature representation:")

    context = PaninianFeatureBridge.to_context(parsed)

    print(
        json.dumps(
            context,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nSelf-test:")
    self_test()
    print("  PASS")


# ============================================================================
# 16. Self-test
# ============================================================================

def self_test() -> None:
    parser = MarathiParser()

    # Unicode normalization.
    text = "मी\u200b पुस्तक वाचतो."
    normalized = MarathiNormalizer.normalize(text)

    assert "\u200b" not in normalized

    # Tokenization.
    tokens = MarathiTokenizer.tokenize("राम घरात आहे.")
    assert [token.text for token in tokens] == [
        "राम",
        "घरात",
        "आहे",
        ".",
    ]

    # Exact lexicon lookup.
    parsed = parser.parse("राम आहे.")

    ram = parsed.best_analysis(0)
    assert ram is not None
    assert ram.features.lemma == "राम"
    assert ram.features.pos == PartOfSpeech.NOUN
    assert ram.features.gender == Gender.MASCULINE

    # Verb lookup.
    verb = parsed.best_analysis(1)
    assert verb is not None
    assert verb.features.pos == PartOfSpeech.VERB
    assert verb.features.tense == Tense.PRESENT

    # Unknown word remains explicit rather than being silently discarded.
    unknown = parser.parse("झगमगाटxyz आहे.")
    unknown_analysis = unknown.best_analysis(0)

    assert unknown_analysis is not None
    assert unknown_analysis.features.pos == PartOfSpeech.UNKNOWN

    # Ambiguity is preserved.
    lexicon = MarathiLexicon()

    lexicon.add(
        LexicalEntry(
            surface="कर",
            lemma="कर",
            pos=PartOfSpeech.NOUN,
        )
    )

    lexicon.add(
        LexicalEntry(
            surface="कर",
            lemma="कर",
            pos=PartOfSpeech.VERB,
        )
    )

    ambiguous_parser = MarathiParser(
        lexicon=lexicon,
        suffix_rules=[],
    )

    ambiguous = ambiguous_parser.parse("कर.")

    candidates = ambiguous.analyses[0]

    assert len(candidates) == 2
    assert {
        candidate.features.pos
        for candidate in candidates
    } == {
        PartOfSpeech.NOUN,
        PartOfSpeech.VERB,
    }

    # Paninian bridge.
    context = parse_to_paninian_context("राम आहे.")

    assert context["language"] == "mr"
    assert context["script"] == "Devanagari"
    assert context["token_count"] == 3

    # JSON serialization.
    json_payload = parsed.to_json()

    assert "राम" in json_payload
    assert "features" in json_payload

    # Batch parsing.
    batch = parser.parse_many(
        [
            "राम आहे.",
            "सीता गेली.",
        ]
    )

    assert len(batch) == 2


if __name__ == "__main__":
    demo()
