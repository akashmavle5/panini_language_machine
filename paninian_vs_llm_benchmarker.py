"""
paninian_vs_llm_benchmarker.py
==============================

Panini Language Machine — Benchmarking Framework

File 12/14.

Purpose
-------
Compare a baseline language-model attention mechanism with the
Paninian/neuro-symbolic attention mechanism produced by Files 9-11.

The benchmark is designed around the hypothesis:

    deterministic linguistic structure
             +
    neural representation learning
             ↓
    better sample efficiency / structural fidelity

This file is an experiment harness, not a claim that Paninian structure
already improves a pretrained LLM.

It supports four levels of comparison:

1. Structural attention
       raw logits vs logits + symbolic bias

2. Synthetic prediction
       baseline linear model vs symbolic-gated model

3. Representation fidelity
       how strongly attention follows known symbolic relations

4. Training efficiency
       epochs / loss / symbolic adherence / parameter count

The benchmark intentionally has a dependency-light reference implementation.
PyTorch is optional and is not required for the core experiments.

The benchmark can later be connected to an actual Hugging Face/PyTorch model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# ============================================================================
# 1. Data structures
# ============================================================================

@dataclass
class BenchmarkConfig:
    seed: int = 42
    epochs: int = 30
    learning_rate: float = 0.08
    symbolic_scale: float = 1.0
    hidden_size: int = 8
    samples: int = 300
    test_fraction: float = 0.25
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkExample:
    """
    Minimal synthetic sequence example.

    token_count:
        Number of tokens.

    relations:
        Directed structural edges:
            source -> target

    target:
        Token that should receive maximum attention.

    structural_tokens:
        Tokens considered structurally relevant to the target.
    """

    token_count: int
    relations: List[Tuple[int, int]]
    target: int
    structural_tokens: List[int]
    difficulty: str = "standard"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttentionMetrics:
    target_accuracy: float
    structural_adherence: float
    mean_entropy: float
    mean_target_attention: float
    mean_structural_attention: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingResult:
    model_name: str
    initial_loss: float
    final_loss: float
    best_loss: float
    final_accuracy: float
    final_structural_adherence: float
    epochs_run: int
    elapsed_seconds: float
    loss_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    config: Dict[str, Any]
    attention_baseline: AttentionMetrics
    attention_paninian: AttentionMetrics
    training_baseline: TrainingResult
    training_paninian: TrainingResult
    relative_improvement: Dict[str, float]
    conclusion: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# 2. Numerical utilities
# ============================================================================

def softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []

    maximum = max(values)
    exponentials = [
        math.exp(float(value) - maximum)
        for value in values
    ]
    total = sum(exponentials)

    if total <= 0:
        return [1.0 / len(values)] * len(values)

    return [
        value / total
        for value in exponentials
    ]


def cross_entropy(
    probabilities: Sequence[float],
    target: int,
) -> float:
    if not probabilities:
        return 0.0

    probability = max(
        min(float(probabilities[target]), 1.0 - 1e-12),
        1e-12,
    )

    return -math.log(probability)


def entropy(
    probabilities: Sequence[float],
) -> float:
    result = 0.0

    for probability in probabilities:
        if probability > 0:
            result -= probability * math.log(
                probability,
                2,
            )

    return result


def argmax(
    values: Sequence[float],
) -> int:
    if not values:
        raise ValueError("Cannot argmax an empty sequence.")

    return max(
        range(len(values)),
        key=lambda index: values[index],
    )


def mean_or_zero(
    values: Sequence[float],
) -> float:
    return (
        statistics.fmean(values)
        if values
        else 0.0
    )


# ============================================================================
# 3. Synthetic structural dataset
# ============================================================================

def make_symbolic_bias(
    token_count: int,
    relations: Sequence[Tuple[int, int]],
    scale: float = 1.0,
    reverse_scale: float = 0.50,
) -> List[List[float]]:
    matrix = [
        [0.0 for _ in range(token_count)]
        for _ in range(token_count)
    ]

    for source, target in relations:
        if not (
            0 <= source < token_count
            and 0 <= target < token_count
        ):
            continue

        matrix[source][target] += 4.0 * scale
        matrix[target][source] += (
            2.0 * scale * reverse_scale
        )

    return matrix


def generate_example(
    rng: random.Random,
    difficulty: str = "standard",
) -> BenchmarkExample:
    """
    Generate a small dependency-like structure.

    The task is deliberately synthetic so that benchmark improvements can
    initially be attributed to structural information rather than external
    corpora.
    """

    token_count = rng.choice(
        [5, 6, 7, 8]
    )

    predicate = rng.randrange(
        1,
        token_count - 1,
    )

    subject_candidates = [
        index
        for index in range(predicate)
    ]

    object_candidates = [
        index
        for index in range(
            predicate + 1,
            token_count,
        )
    ]

    subject = rng.choice(
        subject_candidates
    )

    object_index = rng.choice(
        object_candidates
    )

    relations = [
        (subject, predicate),
        (object_index, predicate),
    ]

    structural_tokens = [
        subject,
        object_index,
        predicate,
    ]

    # The prediction target is the predicate for this task.
    target = predicate

    if difficulty == "hard":
        # Add an irrelevant distractor relation.
        distractor_source = rng.randrange(
            token_count
        )

        if distractor_source not in structural_tokens:
            relations.append(
                (
                    distractor_source,
                    rng.randrange(
                        token_count
                    ),
                )
            )

    return BenchmarkExample(
        token_count=token_count,
        relations=relations,
        target=target,
        structural_tokens=structural_tokens,
        difficulty=difficulty,
    )


def generate_dataset(
    samples: int,
    seed: int = 42,
    difficulty: str = "standard",
) -> List[BenchmarkExample]:
    rng = random.Random(seed)

    return [
        generate_example(
            rng,
            difficulty=difficulty,
        )
        for _ in range(samples)
    ]


# ============================================================================
# 4. Attention benchmark
# ============================================================================

def random_logits(
    token_count: int,
    rng: random.Random,
    noise: float = 1.0,
) -> List[List[float]]:
    return [
        [
            rng.gauss(
                0.0,
                noise,
            )
            for _ in range(token_count)
        ]
        for _ in range(token_count)
    ]


def add_matrices(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> List[List[float]]:
    if len(left) != len(right):
        raise ValueError(
            "Matrix row counts do not match."
        )

    return [
        [
            float(a) + float(b)
            for a, b in zip(
                left_row,
                right_row,
            )
        ]
        for left_row, right_row in zip(
            left,
            right,
        )
    ]


def attention_matrix(
    logits: Sequence[Sequence[float]],
) -> List[List[float]]:
    return [
        softmax(row)
        for row in logits
    ]


def attention_metrics(
    examples: Sequence[BenchmarkExample],
    symbolic: bool,
    symbolic_scale: float,
    seed: int,
) -> AttentionMetrics:
    rng = random.Random(seed)

    accuracies = []
    adherence = []
    entropies = []
    target_attention = []
    structural_attention = []

    for example in examples:
        logits = random_logits(
            example.token_count,
            rng,
            noise=1.25,
        )

        if symbolic:
            bias = make_symbolic_bias(
                example.token_count,
                example.relations,
                scale=symbolic_scale,
            )

            logits = add_matrices(
                logits,
                bias,
            )

        attention = attention_matrix(
            logits
        )

        # Examine the target's incoming attention row.
        row = attention[
            example.target
        ]

        prediction = argmax(row)

        accuracies.append(
            float(
                prediction
                in example.structural_tokens
            )
        )

        structural_mass = sum(
            row[index]
            for index in example.structural_tokens
            if 0 <= index < len(row)
        )

        adherence.append(
            structural_mass
        )

        target_attention.append(
            row[example.target]
        )

        structural_attention.append(
            structural_mass
        )

        entropies.append(
            entropy(row)
        )

    return AttentionMetrics(
        target_accuracy=mean_or_zero(
            accuracies
        ),
        structural_adherence=mean_or_zero(
            adherence
        ),
        mean_entropy=mean_or_zero(
            entropies
        ),
        mean_target_attention=mean_or_zero(
            target_attention
        ),
        mean_structural_attention=mean_or_zero(
            structural_attention
        ),
    )


# ============================================================================
# 5. Reference training models
# ============================================================================

class ScalarAttentionModel:
    """
    A deliberately tiny model.

    The model learns a single scalar controlling how much structural evidence
    influences the output.

    This is not an LLM. It is a controlled experiment isolating the
    contribution of symbolic structure.
    """

    def __init__(
        self,
        symbolic_enabled: bool,
        learning_rate: float,
    ) -> None:
        self.symbolic_enabled = symbolic_enabled
        self.learning_rate = learning_rate

        # Learned scalar.
        self.weight = 0.0

    def score(
        self,
        example: BenchmarkExample,
        symbolic_scale: float,
    ) -> List[float]:
        """
        Produce token-level logits for the target prediction.

        Baseline receives only weak positional evidence.
        Paninian model receives learned structural evidence.
        """

        scores = [
            0.0
            for _ in range(
                example.token_count
            )
        ]

        # Weak baseline signal.
        for index in range(
            example.token_count
        ):
            distance = abs(
                index - example.target
            )

            scores[index] += (
                -0.08 * distance
            )

        if self.symbolic_enabled:
            for source, target in example.relations:
                if target != example.target:
                    continue

                if 0 <= source < example.token_count:
                    scores[source] += (
                        self.weight
                        * symbolic_scale
                    )

        # The target is itself structurally relevant.
        scores[example.target] += (
            self.weight
            if self.symbolic_enabled
            else 0.0
        )

        return scores

    def probabilities(
        self,
        example: BenchmarkExample,
        symbolic_scale: float,
    ) -> List[float]:
        return softmax(
            self.score(
                example,
                symbolic_scale,
            )
        )

    def train_step(
        self,
        example: BenchmarkExample,
        symbolic_scale: float,
    ) -> float:
        probabilities = self.probabilities(
            example,
            symbolic_scale,
        )

        loss = cross_entropy(
            probabilities,
            example.target,
        )

        if not self.symbolic_enabled:
            return loss

        # Approximate gradient for the scalar parameter.
        #
        # We intentionally use an explicit finite difference gradient instead
        # of hiding the experiment inside a framework.
        epsilon = 1e-4

        original = self.weight

        self.weight = (
            original + epsilon
        )

        plus = cross_entropy(
            self.probabilities(
                example,
                symbolic_scale,
            ),
            example.target,
        )

        self.weight = (
            original - epsilon
        )

        minus = cross_entropy(
            self.probabilities(
                example,
                symbolic_scale,
            ),
            example.target,
        )

        self.weight = original

        gradient = (
            plus - minus
        ) / (
            2.0 * epsilon
        )

        self.weight -= (
            self.learning_rate
            * gradient
        )

        # Keep the experiment numerically stable.
        self.weight = max(
            min(self.weight, 20.0),
            -20.0,
        )

        return loss


# ============================================================================
# 6. Training benchmark
# ============================================================================

def evaluate_model(
    model: ScalarAttentionModel,
    examples: Sequence[BenchmarkExample],
    symbolic_scale: float,
) -> Tuple[float, float]:
    losses = []
    correct = []
    adherence = []

    for example in examples:
        probabilities = model.probabilities(
            example,
            symbolic_scale,
        )

        losses.append(
            cross_entropy(
                probabilities,
                example.target,
            )
        )

        prediction = argmax(
            probabilities
        )

        correct.append(
            float(
                prediction
                == example.target
            )
        )

        structural_mass = sum(
            probabilities[index]
            for index in example.structural_tokens
        )

        adherence.append(
            structural_mass
        )

    return (
        mean_or_zero(losses),
        mean_or_zero(correct),
        mean_or_zero(adherence),
    )


def train_model(
    name: str,
    model: ScalarAttentionModel,
    train_examples: Sequence[BenchmarkExample],
    test_examples: Sequence[BenchmarkExample],
    config: BenchmarkConfig,
) -> TrainingResult:
    started = time.perf_counter()

    initial_loss, _, _ = evaluate_model(
        model,
        test_examples,
        config.symbolic_scale,
    )

    history = []
    best_loss = float("inf")

    for epoch in range(
        config.epochs
    ):
        shuffled = list(
            train_examples
        )

        random.Random(
            config.seed + epoch
        ).shuffle(shuffled)

        for example in shuffled:
            model.train_step(
                example,
                config.symbolic_scale,
            )

        test_loss, _, _ = evaluate_model(
            model,
            test_examples,
            config.symbolic_scale,
        )

        history.append(
            test_loss
        )

        best_loss = min(
            best_loss,
            test_loss,
        )

        if config.verbose:
            print(
                f"{name} "
                f"epoch={epoch + 1:03d} "
                f"loss={test_loss:.5f}"
            )

    final_loss, accuracy, adherence = (
        evaluate_model(
            model,
            test_examples,
            config.symbolic_scale,
        )
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    return TrainingResult(
        model_name=name,
        initial_loss=initial_loss,
        final_loss=final_loss,
        best_loss=best_loss,
        final_accuracy=accuracy,
        final_structural_adherence=adherence,
        epochs_run=config.epochs,
        elapsed_seconds=elapsed,
        loss_history=history,
    )


# ============================================================================
# 7. Relative comparison
# ============================================================================

def relative_delta(
    baseline: float,
    improved: float,
) -> float:
    denominator = abs(baseline)

    if denominator < 1e-12:
        return 0.0

    return (
        (improved - baseline)
        / denominator
    ) * 100.0


def build_conclusion(
    baseline_attention: AttentionMetrics,
    paninian_attention: AttentionMetrics,
    baseline_training: TrainingResult,
    paninian_training: TrainingResult,
) -> str:
    attention_gain = (
        paninian_attention.structural_adherence
        - baseline_attention.structural_adherence
    )

    accuracy_gain = (
        paninian_training.final_accuracy
        - baseline_training.final_accuracy
    )

    if (
        attention_gain > 0.05
        and accuracy_gain > 0.05
    ):
        return (
            "In this controlled synthetic benchmark, "
            "Paninian structural bias improves both "
            "structural attention adherence and target "
            "prediction accuracy. This supports further "
            "testing with real linguistic datasets, but "
            "does not establish superiority over LLMs."
        )

    if attention_gain > 0.05:
        return (
            "Paninian structural bias substantially changes "
            "attention toward structurally relevant tokens, "
            "but the current controlled training experiment "
            "does not establish a corresponding accuracy gain."
        )

    return (
        "The current synthetic benchmark does not show a "
        "clear advantage. The hypothesis should be tested "
        "with larger and linguistically realistic datasets."
    )


# ============================================================================
# 8. Full benchmark
# ============================================================================

def run_benchmark(
    config: Optional[BenchmarkConfig] = None,
) -> BenchmarkReport:
    config = config or BenchmarkConfig()

    dataset = generate_dataset(
        samples=config.samples,
        seed=config.seed,
        difficulty="standard",
    )

    split = max(
        1,
        int(
            len(dataset)
            * (1.0 - config.test_fraction)
        ),
    )

    train_examples = dataset[:split]
    test_examples = dataset[split:]

    if not test_examples:
        test_examples = train_examples

    baseline_attention = attention_metrics(
        test_examples,
        symbolic=False,
        symbolic_scale=config.symbolic_scale,
        seed=config.seed + 100,
    )

    paninian_attention = attention_metrics(
        test_examples,
        symbolic=True,
        symbolic_scale=config.symbolic_scale,
        seed=config.seed + 100,
    )

    baseline_model = ScalarAttentionModel(
        symbolic_enabled=False,
        learning_rate=config.learning_rate,
    )

    paninian_model = ScalarAttentionModel(
        symbolic_enabled=True,
        learning_rate=config.learning_rate,
    )

    baseline_training = train_model(
        "baseline",
        baseline_model,
        train_examples,
        test_examples,
        config,
    )

    paninian_training = train_model(
        "paninian",
        paninian_model,
        train_examples,
        test_examples,
        config,
    )

    relative_improvement = {
        "attention_structural_adherence_pct": relative_delta(
            baseline_attention.structural_adherence,
            paninian_attention.structural_adherence,
        ),
        "attention_target_mass_pct": relative_delta(
            baseline_attention.mean_target_attention,
            paninian_attention.mean_target_attention,
        ),
        "training_accuracy_pct": relative_delta(
            baseline_training.final_accuracy,
            paninian_training.final_accuracy,
        ),
        "training_structural_adherence_pct": relative_delta(
            baseline_training.final_structural_adherence,
            paninian_training.final_structural_adherence,
        ),
        "loss_change_pct": relative_delta(
            baseline_training.final_loss,
            paninian_training.final_loss,
        ),
    }

    conclusion = build_conclusion(
        baseline_attention,
        paninian_attention,
        baseline_training,
        paninian_training,
    )

    return BenchmarkReport(
        config=config.to_dict(),
        attention_baseline=baseline_attention,
        attention_paninian=paninian_attention,
        training_baseline=baseline_training,
        training_paninian=paninian_training,
        relative_improvement=relative_improvement,
        conclusion=conclusion,
    )


# ============================================================================
# 9. Real-model integration specification
# ============================================================================

def transformer_integration_spec() -> Dict[str, Any]:
    """
    Contract for connecting this benchmark to a real Transformer.

    No specific model family is assumed.
    """

    return {
        "baseline": {
            "attention": "QK^T / sqrt(d)",
            "softmax": "standard",
            "symbolic_input": False,
        },
        "paninian": {
            "attention": (
                "QK^T / sqrt(d) + alpha * symbolic_bias"
            ),
            "softmax": "standard",
            "symbolic_input": True,
            "alpha": (
                "learned or configurable symbolic gate"
            ),
        },
        "recommended_loss": {
            "language_model_loss": "cross_entropy",
            "symbolic_consistency_loss": (
                "KL/contrastive/edge-ranking candidate"
            ),
            "total": (
                "L_lm + lambda * L_symbolic"
            ),
        },
        "metrics": [
            "validation_loss",
            "perplexity",
            "token_accuracy",
            "structural_adherence",
            "attention_entropy",
            "long_distance_dependency_accuracy",
            "sample_efficiency",
            "training_tokens",
            "training_time",
            "GPU_memory",
        ],
    }


# ============================================================================
# 10. Report rendering
# ============================================================================

def render_report(
    report: BenchmarkReport,
) -> str:
    b_att = report.attention_baseline
    p_att = report.attention_paninian

    b_train = report.training_baseline
    p_train = report.training_paninian

    lines = [
        "=" * 104,
        "PANINIAN VS STANDARD LLM BENCHMARK",
        "=" * 104,
        "",
        "CONFIGURATION",
        "-" * 104,
    ]

    for key, value in report.config.items():
        lines.append(
            f"{key:<24}: {value}"
        )

    lines.extend([
        "",
        "ATTENTION BENCHMARK",
        "-" * 104,
        f"{'Metric':<36}"
        f"{'Baseline':>18}"
        f"{'Paninian':>18}",
        "-" * 104,
        f"{'Structural adherence':<36}"
        f"{b_att.structural_adherence:>18.4f}"
        f"{p_att.structural_adherence:>18.4f}",
        f"{'Target attention':<36}"
        f"{b_att.mean_target_attention:>18.4f}"
        f"{p_att.mean_target_attention:>18.4f}",
        f"{'Target accuracy':<36}"
        f"{b_att.target_accuracy:>18.4f}"
        f"{p_att.target_accuracy:>18.4f}",
        f"{'Attention entropy':<36}"
        f"{b_att.mean_entropy:>18.4f}"
        f"{p_att.mean_entropy:>18.4f}",
        "",
        "TRAINING BENCHMARK",
        "-" * 104,
        f"{'Metric':<36}"
        f"{'Baseline':>18}"
        f"{'Paninian':>18}",
        "-" * 104,
        f"{'Initial loss':<36}"
        f"{b_train.initial_loss:>18.5f}"
        f"{p_train.initial_loss:>18.5f}",
        f"{'Final loss':<36}"
        f"{b_train.final_loss:>18.5f}"
        f"{p_train.final_loss:>18.5f}",
        f"{'Best loss':<36}"
        f"{b_train.best_loss:>18.5f}"
        f"{p_train.best_loss:>18.5f}",
        f"{'Final accuracy':<36}"
        f"{b_train.final_accuracy:>18.4f}"
        f"{p_train.final_accuracy:>18.4f}",
        f"{'Structural adherence':<36}"
        f"{b_train.final_structural_adherence:>18.4f}"
        f"{p_train.final_structural_adherence:>18.4f}",
        f"{'Elapsed seconds':<36}"
        f"{b_train.elapsed_seconds:>18.4f}"
        f"{p_train.elapsed_seconds:>18.4f}",
        "",
        "RELATIVE CHANGE: PANINIAN vs BASELINE",
        "-" * 104,
    ])

    for key, value in report.relative_improvement.items():
        lines.append(
            f"{key:<48}: {value:>10.2f}%"
        )

    lines.extend([
        "",
        "INTERPRETATION",
        "-" * 104,
        report.conclusion,
        "",
        "IMPORTANT:",
        "This benchmark is controlled and synthetic.",
        "It demonstrates the experimental mechanism, not general LLM superiority.",
        "Real claims require matched datasets, identical compute budgets,",
        "multiple random seeds, real Transformer models, and statistical testing.",
    ])

    return "\n".join(lines)


# ============================================================================
# 11. JSON export
# ============================================================================

def write_report(
    report: BenchmarkReport,
    path: str | Path,
) -> Path:
    destination = Path(path)

    destination.write_text(
        json.dumps(
            report.to_dict(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return destination


# ============================================================================
# 12. Self-test
# ============================================================================

def self_test() -> None:
    config = BenchmarkConfig(
        seed=7,
        epochs=5,
        learning_rate=0.08,
        symbolic_scale=1.0,
        samples=40,
        test_fraction=0.25,
    )

    dataset = generate_dataset(
        samples=40,
        seed=7,
    )

    assert len(dataset) == 40

    example = dataset[0]

    assert example.token_count >= 5
    assert example.target in range(
        example.token_count
    )

    bias = make_symbolic_bias(
        example.token_count,
        example.relations,
    )

    assert len(bias) == example.token_count
    assert all(
        len(row) == example.token_count
        for row in bias
    )

    # Bias should be non-zero if relations exist.
    assert any(
        value != 0.0
        for row in bias
        for value in row
    )

    logits = [
        [0.0, 0.0],
        [0.0, 0.0],
    ]

    attention = attention_matrix(
        logits
    )

    assert all(
        abs(sum(row) - 1.0) < 1e-9
        for row in attention
    )

    symbolic = symbolic_attention_test(
        logits,
        [
            [0.0, 5.0],
            [2.5, 0.0],
        ],
    )

    assert symbolic[0][1] > symbolic[0][0]
    assert symbolic[1][0] > symbolic[1][1]

    report = run_benchmark(
        config
    )

    assert (
        report.training_baseline.epochs_run
        == config.epochs
    )

    assert (
        report.training_paninian.epochs_run
        == config.epochs
    )

    assert report.relative_improvement

    serialized = json.dumps(
        report.to_dict()
    )

    assert "baseline" in serialized
    assert "paninian" in serialized
    assert "structural_adherence" in serialized

    spec = transformer_integration_spec()

    assert (
        "QK^T / sqrt(d) + alpha * symbolic_bias"
        in spec["paninian"]["attention"]
    )


def symbolic_attention_test(
    logits: Sequence[Sequence[float]],
    bias: Sequence[Sequence[float]],
) -> List[List[float]]:
    combined = add_matrices(
        logits,
        bias,
    )

    return attention_matrix(
        combined
    )


# ============================================================================
# 13. CLI
# ============================================================================

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paninian_vs_llm_benchmarker",
        description=(
            "Compare baseline and Paninian structural attention "
            "in a controlled benchmark."
        ),
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.08,
    )

    parser.add_argument(
        "--symbolic-scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Write benchmark report to JSON.",
    )

    parser.add_argument(
        "--architecture",
        action="store_true",
        help="Print real Transformer integration specification.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    return parser


def main(
    argv: Optional[Sequence[str]] = None,
) -> int:
    args = build_cli().parse_args(argv)

    if args.self_test:
        self_test()
        print("SELF_TEST_PASS")
        return 0

    if args.architecture:
        print(
            json.dumps(
                transformer_integration_spec(),
                indent=2,
                ensure_ascii=False,
            )
        )

        if not args.samples:
            return 0

    config = BenchmarkConfig(
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        symbolic_scale=args.symbolic_scale,
        samples=args.samples,
        test_fraction=args.test_fraction,
        verbose=args.verbose,
    )

    report = run_benchmark(
        config
    )

    print(
        render_report(report)
    )

    if args.json:
        write_report(
            report,
            args.json,
        )
        print(
            f"\nReport written to: {args.json}"
        )

    return 0


if __name__ == "__main__":
    main()
