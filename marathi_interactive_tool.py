"""
marathi_interactive_tool.py
===========================

Panini Language Machine — Interactive Marathi/Pāṇinian Analysis Tool

File 6/14.

Purpose
-------
Expose `marathi_parser.py` as an interactive command-line laboratory for
linguistic experimentation.

The tool provides:

    sentence input
        ↓
    MarathiParser
        ↓
    token/morphology display
        ↓
    ambiguity inspection
        ↓
    Paninian feature representation
        ↓
    JSON export

This file deliberately remains a thin presentation/orchestration layer.
The linguistic logic lives in `marathi_parser.py`.

Supported modes
---------------
    interactive
    parse
    tokens
    morphology
    context
    json
    batch

The command-line interface is implemented with the Python standard library
so that the prototype can run without installing a web framework.

Design principle
----------------
Keep analysis inspectable.

A researcher should be able to see:
    * what the tokenizer produced
    * which lexical entries matched
    * which suffix hypotheses fired
    * what grammatical features were assigned
    * where ambiguity remains
    * what symbolic representation is handed to later Paninian layers

This is an experimentation interface, not a claim of complete Marathi
grammar coverage.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from marathi_parser import (
    MarathiParser,
    MarathiTokenizer,
    MorphAnalysis,
    ParsedSentence,
    PaninianFeatureBridge,
    PartOfSpeech,
    build_prototype_marathi_lexicon,
    build_prototype_suffix_rules,
    parse_marathi,
)


# ============================================================================
# 1. Output formatting
# ============================================================================

@dataclass(frozen=True)
class DisplayOptions:
    show_scores: bool = True
    show_evidence: bool = True
    show_unknown: bool = True
    max_candidates: int = 10
    unicode: bool = True


class MarathiDisplay:
    """
    Human-readable presentation layer.

    No linguistic inference occurs here.
    """

    WIDTH = 96

    @staticmethod
    def header(title: str) -> str:
        line = "=" * MarathiDisplay.WIDTH
        return f"\n{line}\n{title}\n{line}"

    @staticmethod
    def token_table(
        parsed: ParsedSentence,
        options: DisplayOptions,
    ) -> str:
        lines = [
            MarathiDisplay.header("TOKENS"),
            f"{'IDX':>3}  {'SURFACE':<18} {'KIND':<12} {'SPAN':<12}",
            "-" * 60,
        ]

        for token in parsed.tokens:
            lines.append(
                f"{token.index:>3}  "
                f"{token.text:<18} "
                f"{token.kind.value:<12} "
                f"{token.start}:{token.end:<7}"
            )

        return "\n".join(lines)

    @staticmethod
    def _feature_value(value: Any) -> str:
        if hasattr(value, "value"):
            return str(value.value)

        if value is None:
            return "-"

        if isinstance(value, dict):
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
            )

        return str(value)

    @staticmethod
    def analysis_line(
        analysis: MorphAnalysis,
        rank: int,
        options: DisplayOptions,
    ) -> str:
        f = analysis.features

        parts = [
            f"    {rank}.",
            f"POS={MarathiDisplay._feature_value(f.pos)}",
            f"lemma={MarathiDisplay._feature_value(f.lemma)}",
            f"gender={MarathiDisplay._feature_value(f.gender)}",
            f"number={MarathiDisplay._feature_value(f.number)}",
            f"case={MarathiDisplay._feature_value(f.case)}",
            f"tense={MarathiDisplay._feature_value(f.tense)}",
            f"aspect={MarathiDisplay._feature_value(f.aspect)}",
            f"source={analysis.source}",
        ]

        if options.show_scores:
            parts.append(f"score={analysis.score:.3f}")

        return " ".join(parts)

    @staticmethod
    def morphology(
        parsed: ParsedSentence,
        options: DisplayOptions,
    ) -> str:
        lines = [
            MarathiDisplay.header("MORPHOLOGICAL ANALYSIS")
        ]

        for token in parsed.tokens:
            candidates = parsed.analyses.get(token.index, [])

            if (
                not options.show_unknown
                and candidates
                and candidates[0].features.pos == PartOfSpeech.UNKNOWN
            ):
                continue

            lines.append(
                f"\n[{token.index}] {token.text}"
            )

            if not candidates:
                lines.append("    No analysis")
                continue

            for rank, candidate in enumerate(
                candidates[: options.max_candidates],
                start=1,
            ):
                lines.append(
                    MarathiDisplay.analysis_line(
                        candidate,
                        rank,
                        options,
                    )
                )

                if options.show_evidence:
                    for evidence in candidate.evidence:
                        lines.append(f"       evidence: {evidence}")

            if len(candidates) > options.max_candidates:
                lines.append(
                    f"    ... {len(candidates) - options.max_candidates} "
                    "additional candidates omitted"
                )

        return "\n".join(lines)

    @staticmethod
    def ambiguity(
        parsed: ParsedSentence,
    ) -> str:
        lines = [
            MarathiDisplay.header("AMBIGUITY REPORT")
        ]

        ambiguous_count = 0

        for token in parsed.tokens:
            candidates = parsed.analyses.get(token.index, [])

            if len(candidates) <= 1:
                continue

            ambiguous_count += 1

            pos_values = sorted(
                {
                    MarathiDisplay._feature_value(
                        candidate.features.pos
                    )
                    for candidate in candidates
                }
            )

            lemma_values = sorted(
                {
                    MarathiDisplay._feature_value(
                        candidate.features.lemma
                    )
                    for candidate in candidates
                }
            )

            lines.append(
                f"[{token.index}] {token.text}: "
                f"{len(candidates)} candidates"
            )
            lines.append(
                f"    POS hypotheses: {', '.join(pos_values)}"
            )
            lines.append(
                f"    Lemma hypotheses: {', '.join(lemma_values)}"
            )

        if ambiguous_count == 0:
            lines.append("No multi-analysis tokens detected.")

        return "\n".join(lines)

    @staticmethod
    def paninian_context(
        parsed: ParsedSentence,
    ) -> str:
        context = PaninianFeatureBridge.to_context(parsed)

        return (
            MarathiDisplay.header("PANINIAN FEATURE CONTEXT")
            + "\n"
            + json.dumps(
                context,
                ensure_ascii=False,
                indent=2,
            )
        )

    @staticmethod
    def trace(
        parsed: ParsedSentence,
    ) -> str:
        lines = [
            MarathiDisplay.header("PARSER TRACE")
        ]

        for index, item in enumerate(parsed.traces):
            token_text = ""

            if item.token_index is not None:
                if 0 <= item.token_index < len(parsed.tokens):
                    token_text = (
                        f" token={parsed.tokens[item.token_index].text}"
                    )

            lines.append(
                f"{index + 1:>3}. "
                f"[{item.stage}] "
                f"{item.message}{token_text}"
            )

            if item.data:
                lines.append(
                    "     "
                    + json.dumps(
                        item.data,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )

        return "\n".join(lines)

    @staticmethod
    def summary(
        parsed: ParsedSentence,
    ) -> str:
        word_tokens = [
            token
            for token in parsed.tokens
            if token.kind.value == "word"
        ]

        ambiguous = sum(
            1
            for token in word_tokens
            if len(parsed.analyses.get(token.index, [])) > 1
        )

        unknown = sum(
            1
            for token in word_tokens
            if (
                parsed.best_analysis(token.index) is not None
                and parsed.best_analysis(token.index).features.pos
                == PartOfSpeech.UNKNOWN
            )
        )

        return "\n".join(
            [
                MarathiDisplay.header("PARSE SUMMARY"),
                f"Sentence       : {parsed.text}",
                f"Tokens         : {len(parsed.tokens)}",
                f"Word tokens    : {len(word_tokens)}",
                f"Ambiguous      : {ambiguous}",
                f"Unknown words  : {unknown}",
            ]
        )


# ============================================================================
# 2. Analysis session
# ============================================================================

class MarathiAnalysisSession:
    """
    Stateful analysis session.

    The session stores the most recent parse and makes it available to
    interactive commands.
    """

    def __init__(
        self,
        parser: Optional[MarathiParser] = None,
    ) -> None:
        self.parser = parser or MarathiParser()
        self.last_parse: Optional[ParsedSentence] = None

    def parse(self, text: str) -> ParsedSentence:
        self.last_parse = self.parser.parse(text)
        return self.last_parse

    def require_parse(self) -> ParsedSentence:
        if self.last_parse is None:
            raise RuntimeError(
                "No sentence has been parsed in this session."
            )

        return self.last_parse


# ============================================================================
# 3. Command operations
# ============================================================================

def parse_sentence(
    session: MarathiAnalysisSession,
    text: str,
) -> ParsedSentence:
    return session.parse(text)


def command_parse(
    session: MarathiAnalysisSession,
    text: str,
    options: DisplayOptions,
) -> int:
    parsed = parse_sentence(session, text)

    print(MarathiDisplay.summary(parsed))
    print(MarathiDisplay.token_table(parsed, options))
    print(MarathiDisplay.morphology(parsed, options))

    return 0


def command_tokens(
    session: MarathiAnalysisSession,
    text: str,
    options: DisplayOptions,
) -> int:
    parsed = parse_sentence(session, text)

    print(MarathiDisplay.token_table(parsed, options))

    return 0


def command_morphology(
    session: MarathiAnalysisSession,
    text: str,
    options: DisplayOptions,
) -> int:
    parsed = parse_sentence(session, text)

    print(MarathiDisplay.morphology(parsed, options))
    print(MarathiDisplay.ambiguity(parsed))

    return 0


def command_context(
    session: MarathiAnalysisSession,
    text: str,
) -> int:
    parsed = parse_sentence(session, text)

    print(MarathiDisplay.paninian_context(parsed))

    return 0


def command_trace(
    session: MarathiAnalysisSession,
    text: str,
) -> int:
    parsed = parse_sentence(session, text)

    print(MarathiDisplay.trace(parsed))

    return 0


def command_json(
    session: MarathiAnalysisSession,
    text: str,
    output: Optional[str] = None,
) -> int:
    parsed = parse_sentence(session, text)

    payload = parsed.to_json()

    if output:
        path = Path(output)
        path.write_text(
            payload,
            encoding="utf-8",
        )

        print(f"JSON written to: {path}")
    else:
        print(payload)

    return 0


def command_batch(
    session: MarathiAnalysisSession,
    input_path: str,
    output_path: Optional[str] = None,
) -> int:
    source = Path(input_path)

    if not source.exists():
        print(
            f"Input file not found: {source}",
            file=sys.stderr,
        )
        return 2

    sentences = [
        line.strip()
        for line in source.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    results = []

    for sentence in sentences:
        parsed = session.parse(sentence)

        results.append(
            {
                "sentence": sentence,
                "analysis": parsed.to_dict(),
                "paninian_context":
                    PaninianFeatureBridge.to_context(parsed),
            }
        )

    payload = json.dumps(
        results,
        ensure_ascii=False,
        indent=2,
    )

    if output_path:
        destination = Path(output_path)
        destination.write_text(
            payload,
            encoding="utf-8",
        )

        print(
            f"Processed {len(results)} sentences."
        )
        print(
            f"Output written to: {destination}"
        )
    else:
        print(payload)

    return 0


# ============================================================================
# 4. Interactive shell
# ============================================================================

class MarathiInteractiveShell:
    """
    Simple REPL.

    Commands:
        :help
        :parse
        :tokens
        :morph
        :ambiguity
        :context
        :trace
        :json
        :summary
        :quit

    Any ordinary line is treated as a Marathi sentence.
    """

    HELP = """
Commands
--------
:help              Show this help
:parse             Parse a new sentence
:tokens            Show tokens for current sentence
:morph             Show morphology for current sentence
:ambiguity         Show ambiguous analyses
:context           Show Paninian feature context
:trace             Show parser trace
:summary           Show parse summary
:json [file]        Export current parse as JSON
:quit / :exit      Exit

Usage
-----
Type a Marathi sentence directly to parse it.

Examples
--------
राम आहे.
सीता गेली.
मी पुस्तक वाचतो.
"""

    def __init__(
        self,
        session: Optional[MarathiAnalysisSession] = None,
        options: Optional[DisplayOptions] = None,
    ) -> None:
        self.session = session or MarathiAnalysisSession()
        self.options = options or DisplayOptions()

    def run(self) -> int:
        print(
            MarathiDisplay.header(
                "PANINI LANGUAGE MACHINE — MARATHI INTERACTIVE TOOL"
            )
        )
        print(
            "Type :help for commands. "
            "Type a Marathi sentence to analyze it."
        )

        while True:
            try:
                line = input("\nmarathi> ").strip()
            except EOFError:
                print()
                return 0
            except KeyboardInterrupt:
                print("\nInterrupted.")
                return 0

            if not line:
                continue

            if line in {":quit", ":exit", ":q"}:
                print("Exiting.")
                return 0

            if line == ":help":
                print(self.HELP)
                continue

            try:
                status = self._dispatch(line)

                if status != 0:
                    return status

            except Exception as exc:
                print(
                    f"Error: {exc}",
                    file=sys.stderr,
                )

    def _dispatch(self, line: str) -> int:
        if line.startswith(":parse"):
            text = line[len(":parse"):].strip()

            if not text:
                text = input("Sentence: ").strip()

            return command_parse(
                self.session,
                text,
                self.options,
            )

        if line.startswith(":tokens"):
            if line[len(":tokens"):].strip():
                text = line[len(":tokens"):].strip()
                self.session.parse(text)

            parsed = self.session.require_parse()
            print(
                MarathiDisplay.token_table(
                    parsed,
                    self.options,
                )
            )
            return 0

        if line.startswith(":morph"):
            if line[len(":morph"):].strip():
                text = line[len(":morph"):].strip()
                self.session.parse(text)

            parsed = self.session.require_parse()

            print(
                MarathiDisplay.morphology(
                    parsed,
                    self.options,
                )
            )
            return 0

        if line.startswith(":ambiguity"):
            parsed = self.session.require_parse()
            print(MarathiDisplay.ambiguity(parsed))
            return 0

        if line.startswith(":context"):
            parsed = self.session.require_parse()
            print(
                MarathiDisplay.paninian_context(parsed)
            )
            return 0

        if line.startswith(":trace"):
            parsed = self.session.require_parse()
            print(MarathiDisplay.trace(parsed))
            return 0

        if line.startswith(":summary"):
            parsed = self.session.require_parse()
            print(MarathiDisplay.summary(parsed))
            return 0

        if line.startswith(":json"):
            output = line[len(":json"):].strip() or None
            parsed = self.session.require_parse()

            return command_json(
                self.session,
                parsed.text,
                output,
            )

        # Default: sentence.
        return command_parse(
            self.session,
            line,
            self.options,
        )


# ============================================================================
# 5. CLI parser
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marathi_interactive_tool",
        description=(
            "Interactive Marathi/Pāṇinian analysis laboratory "
            "built on marathi_parser.py."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    parse_parser = subparsers.add_parser(
        "parse",
        help="Parse a Marathi sentence.",
    )
    parse_parser.add_argument(
        "text",
        help="Marathi sentence.",
    )
    parse_parser.add_argument(
        "--max-candidates",
        type=int,
        default=10,
    )

    tokens_parser = subparsers.add_parser(
        "tokens",
        help="Tokenize a Marathi sentence.",
    )
    tokens_parser.add_argument(
        "text",
    )

    morph_parser = subparsers.add_parser(
        "morphology",
        aliases=["morph"],
        help="Show morphological analyses.",
    )
    morph_parser.add_argument(
        "text",
    )
    morph_parser.add_argument(
        "--max-candidates",
        type=int,
        default=10,
    )

    context_parser = subparsers.add_parser(
        "context",
        help="Output Paninian feature context.",
    )
    context_parser.add_argument(
        "text",
    )

    trace_parser = subparsers.add_parser(
        "trace",
        help="Show parser trace.",
    )
    trace_parser.add_argument(
        "text",
    )

    json_parser = subparsers.add_parser(
        "json",
        help="Export parse as JSON.",
    )
    json_parser.add_argument(
        "text",
    )
    json_parser.add_argument(
        "-o",
        "--output",
    )

    batch_parser = subparsers.add_parser(
        "batch",
        help="Parse one sentence per input line.",
    )
    batch_parser.add_argument(
        "input",
        help="UTF-8 text file containing sentences.",
    )
    batch_parser.add_argument(
        "-o",
        "--output",
    )

    subparsers.add_parser(
        "interactive",
        aliases=["shell", "repl"],
        help="Start interactive shell.",
    )

    return parser


# ============================================================================
# 6. CLI entry point
# ============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    session = MarathiAnalysisSession()

    if args.command in {
        "interactive",
        "shell",
        "repl",
        None,
    }:
        return MarathiInteractiveShell(
            session=session
        ).run()

    if args.command == "parse":
        options = DisplayOptions(
            max_candidates=args.max_candidates
        )

        return command_parse(
            session,
            args.text,
            options,
        )

    if args.command == "tokens":
        return command_tokens(
            session,
            args.text,
            DisplayOptions(),
        )

    if args.command in {"morphology", "morph"}:
        options = DisplayOptions(
            max_candidates=args.max_candidates
        )

        return command_morphology(
            session,
            args.text,
            options,
        )

    if args.command == "context":
        return command_context(
            session,
            args.text,
        )

    if args.command == "trace":
        return command_trace(
            session,
            args.text,
        )

    if args.command == "json":
        return command_json(
            session,
            args.text,
            args.output,
        )

    if args.command == "batch":
        return command_batch(
            session,
            args.input,
            args.output,
        )

    raise RuntimeError(
        f"Unknown command: {args.command}"
    )


# ============================================================================
# 7. Programmatic examples
# ============================================================================

def analyze_sentence(
    text: str,
) -> ParsedSentence:
    """
    Programmatic convenience function.

    This is intentionally equivalent to the parser API and does not add
    additional linguistic assumptions.
    """
    return parse_marathi(text)


def analyze_and_get_context(
    text: str,
) -> dict:
    parsed = analyze_sentence(text)

    return PaninianFeatureBridge.to_context(parsed)


# ============================================================================
# 8. Self-test
# ============================================================================

def self_test() -> None:
    # Basic parse.
    session = MarathiAnalysisSession()

    parsed = session.parse(
        "राम पुस्तक घेऊन गेला."
    )

    assert parsed.text == "राम पुस्तक घेऊन गेला."
    assert len(parsed.tokens) == 5

    # Token command equivalent.
    token_text = MarathiDisplay.token_table(
        parsed,
        DisplayOptions(),
    )

    assert "राम" in token_text
    assert "पुस्तक" in token_text

    # Morphology output.
    morphology_text = MarathiDisplay.morphology(
        parsed,
        DisplayOptions(),
    )

    assert "राम" in morphology_text
    assert "गेला" in morphology_text

    # Ambiguity output should always be valid.
    ambiguity_text = MarathiDisplay.ambiguity(parsed)
    assert "AMBIGUITY REPORT" in ambiguity_text

    # Context bridge.
    context_text = MarathiDisplay.paninian_context(
        parsed
    )

    assert '"language": "mr"' in context_text
    assert '"script": "Devanagari"' in context_text

    # Trace.
    trace_text = MarathiDisplay.trace(parsed)
    assert "normalize" in trace_text
    assert "tokenize" in trace_text
    assert "morphology" in trace_text

    # JSON export.
    payload = parsed.to_json()

    assert "राम" in payload
    assert "features" in payload

    # CLI parser construction.
    cli = build_arg_parser()

    args = cli.parse_args(
        [
            "parse",
            "राम आहे.",
        ]
    )

    assert args.command == "parse"
    assert args.text == "राम आहे."

    # Batch.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        input_file = Path(tmp) / "sentences.txt"
        output_file = Path(tmp) / "results.json"

        input_file.write_text(
            "राम आहे.\nसीता गेली.\n",
            encoding="utf-8",
        )

        status = command_batch(
            session,
            str(input_file),
            str(output_file),
        )

        assert status == 0
        assert output_file.exists()

        results = json.loads(
            output_file.read_text(
                encoding="utf-8"
            )
        )

        assert len(results) == 2

    # Programmatic context.
    context = analyze_and_get_context(
        "मी पुस्तक वाचतो."
    )

    assert context["language"] == "mr"
    assert context["token_count"] == 4


# ============================================================================
# 9. Demo
# ============================================================================

def demo() -> None:
    print(
        MarathiDisplay.header(
            "PANINI LANGUAGE MACHINE — FILE 6 DEMO"
        )
    )

    sentence = "राम पुस्तक घेऊन गेला."

    session = MarathiAnalysisSession()

    parsed = session.parse(sentence)

    print(MarathiDisplay.summary(parsed))
    print(
        MarathiDisplay.token_table(
            parsed,
            DisplayOptions(),
        )
    )
    print(
        MarathiDisplay.morphology(
            parsed,
            DisplayOptions(),
        )
    )
    print(
        MarathiDisplay.ambiguity(parsed)
    )
    print(
        MarathiDisplay.paninian_context(parsed)
    )

    print(
        MarathiDisplay.header("SELF-TEST")
    )

    self_test()

    print("PASS")


if __name__ == "__main__":
    main()
