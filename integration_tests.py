"""
integration_tests.py
====================

Panini Language Machine — End-to-End Integration Test Suite

File 14/14.

Purpose
-------
Validate the complete 14-file Panini Language Machine pipeline:

    Panini Core
        ↓
    Panini Engine
        ↓
    Aṣṭādhyāyī Compiler
        ↓
    Scaled Compiler
        ↓
    Marathi Parser
        ↓
    Interactive / Export Interfaces
        ↓
    Kāraka Dependency Graph
        ↓
    Neuro-Symbolic Panini
        ↓
    Paninian English LLM / VGA
        ↓
    Neuro-Symbolic Trainer
        ↓
    Paninian vs LLM Benchmarker
        ↓
    API Server
        ↓
    System Integration

The tests are deliberately defensive because the project has evolved through
multiple research iterations and earlier files may expose different APIs.

This file therefore validates:

1. File existence
2. Python syntax
3. Importability
4. Public-symbol availability
5. Cross-module contracts where discoverable
6. Benchmark execution
7. API construction
8. Serialization
9. Graph normalization
10. End-to-end system health

The final output is:

    SYSTEM_INTEGRATION_PASS

when all mandatory tests succeed.

A JSON report can also be generated for CI/CD and future experiment tracking.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


# ============================================================================
# 1. Project configuration
# ============================================================================

PROJECT_DIR = Path(__file__).resolve().parent

EXPECTED_FILES = [
    "panini_core.py",
    "panini_engine.py",
    "ashtadhyayi_compiler.py",
    "scaled_panini_compiler.py",
    "marathi_parser.py",
    "marathi_interactive_tool.py",
    "panini_exporter.py",
    "karaka_dependency.py",
    "neuro_symbolic_panini.py",
    "paninian_english_llm.py",
    "neuro_symbolic_trainer.py",
    "paninian_vs_llm_benchmarker.py",
    "panini_api_server.py",
    "integration_tests.py",
]


# ============================================================================
# 2. Test result structures
# ============================================================================

@dataclass
class TestResult:
    name: str
    category: str
    status: str
    duration_seconds: float
    message: str = ""
    details: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IntegrationReport:
    project_dir: str
    started_at: float
    finished_at: float
    passed: int
    failed: int
    skipped: int
    results: List[TestResult]
    system_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_dir": self.project_dir,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": (
                self.finished_at
                - self.started_at
            ),
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "results": [
                item.to_dict()
                for item in self.results
            ],
            "system_status": self.system_status,
        }


# ============================================================================
# 3. Test runner
# ============================================================================

class IntegrationTestRunner:
    def __init__(
        self,
        project_dir: Path = PROJECT_DIR,
        verbose: bool = False,
    ) -> None:
        self.project_dir = Path(
            project_dir
        )
        self.verbose = verbose

        self.results: List[TestResult] = []
        self.modules: Dict[str, Any] = {}
        self.module_errors: Dict[str, str] = {}

    def add(
        self,
        name: str,
        category: str,
        status: str,
        started: float,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        result = TestResult(
            name=name,
            category=category,
            status=status,
            duration_seconds=(
                time.perf_counter()
                - started
            ),
            message=message,
            details=details or {},
        )

        self.results.append(
            result
        )

        if self.verbose:
            print(
                f"[{status}] "
                f"{category}: "
                f"{name}"
            )

            if message:
                print(
                    f"    {message}"
                )

    def run(
        self,
        name: str,
        category: str,
        function,
    ) -> None:
        started = time.perf_counter()

        try:
            details = function()

            self.add(
                name=name,
                category=category,
                status="PASS",
                started=started,
                details=(
                    details
                    if isinstance(
                        details,
                        dict,
                    )
                    else {}
                ),
            )

        except SkipTest as exc:
            self.add(
                name=name,
                category=category,
                status="SKIP",
                started=started,
                message=str(exc),
            )

        except Exception as exc:
            self.add(
                name=name,
                category=category,
                status="FAIL",
                started=started,
                message=(
                    f"{type(exc).__name__}: {exc}"
                ),
                details={
                    "traceback": traceback.format_exc()
                },
            )

    @property
    def passed(self) -> int:
        return sum(
            result.status == "PASS"
            for result in self.results
        )

    @property
    def failed(self) -> int:
        return sum(
            result.status == "FAIL"
            for result in self.results
        )

    @property
    def skipped(self) -> int:
        return sum(
            result.status == "SKIP"
            for result in self.results
        )


class SkipTest(Exception):
    """Explicitly skip a non-mandatory integration test."""


# ============================================================================
# 4. Serialization helper
# ============================================================================

def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if hasattr(
        value,
        "to_dict",
    ) and callable(
        value.to_dict
    ):
        return json_safe(
            value.to_dict()
        )

    if hasattr(
        value,
        "__dataclass_fields__",
    ):
        return json_safe(
            asdict(value)
        )

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            json_safe(item)
            for item in value
        ]

    if hasattr(
        value,
        "__dict__",
    ):
        return {
            str(key): json_safe(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }

    return str(value)


# ============================================================================
# 5. Dynamic import system
# ============================================================================

def load_module(
    filename: str,
    project_dir: Path,
) -> Any:
    path = (
        project_dir
        / filename
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing project file: {path}"
        )

    module_name = (
        "panini_integration_"
        + path.stem.replace(
            "-",
            "_",
        )
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(path),
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise ImportError(
            f"Unable to create import specification for {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    # Critical for dataclasses and reflective libraries.
    sys.modules[module_name] = module

    project_path = str(
        project_dir
    )

    if project_path not in sys.path:
        sys.path.insert(
            0,
            project_path,
        )

    spec.loader.exec_module(
        module
    )

    return module


def load_all_modules(
    runner: IntegrationTestRunner,
) -> Dict[str, Any]:
    for filename in EXPECTED_FILES:
        if filename == "integration_tests.py":
            continue

        try:
            module = load_module(
                filename,
                runner.project_dir,
            )

            runner.modules[
                filename
            ] = module

        except Exception as exc:
            runner.module_errors[
                filename
            ] = (
                f"{type(exc).__name__}: {exc}"
            )

    return runner.modules


# ============================================================================
# 6. File-level tests
# ============================================================================

def test_all_files_exist(
    runner: IntegrationTestRunner,
) -> Dict[str, Any]:
    missing = [
        filename
        for filename in EXPECTED_FILES
        if not (
            runner.project_dir
            / filename
        ).exists()
    ]

    if missing:
        raise AssertionError(
            "Missing files: "
            + ", ".join(missing)
        )

    return {
        "file_count": len(
            EXPECTED_FILES
        ),
        "missing": [],
    }


def test_python_syntax(
    runner: IntegrationTestRunner,
) -> Dict[str, Any]:
    failures = []

    for filename in EXPECTED_FILES:
        path = (
            runner.project_dir
            / filename
        )

        if not path.exists():
            continue

        source = path.read_text(
            encoding="utf-8"
        )

        try:
            ast.parse(
                source,
                filename=str(path),
            )
        except SyntaxError as exc:
            failures.append({
                "file": filename,
                "error": str(exc),
            })

    if failures:
        raise AssertionError(
            json.dumps(
                failures,
                indent=2,
            )
        )

    return {
        "validated_files": len(
            EXPECTED_FILES
        ),
    }


def test_modules_import(
    runner: IntegrationTestRunner,
) -> Dict[str, Any]:
    load_all_modules(
        runner
    )

    failures = dict(
        runner.module_errors
    )

    if failures:
        raise AssertionError(
            json.dumps(
                failures,
                indent=2,
            )
        )

    return {
        "imported_modules": len(
            runner.modules
        ),
    }


# ============================================================================
# 7. Public API discovery
# ============================================================================

def public_symbols(
    module: Any,
) -> List[str]:
    return sorted(
        name
        for name in dir(module)
        if not name.startswith("_")
    )


def test_core_symbols(
    runner: IntegrationTestRunner,
) -> Dict[str, Any]:
    required = {
        "panini_core.py": [
            "Panini",
            "PaniniRule",
            "Rule",
        ],
        "panini_engine.py": [
            "PaniniEngine",
        ],
        "karaka_dependency.py": [
            "Karaka",
        ],
        "paninian_vs_llm_benchmarker.py": [
            "BenchmarkConfig",
            "run_benchmark",
            "transformer_integration_spec",
        ],
        "panini_api_server.py": [
            "architecture",
            "analyze_text",
            "graph_from_analysis",
            "create_server",
        ],
    }

    found = {}
    missing = {}

    for filename, names in required.items():
        module = runner.modules.get(
            filename
        )

        if module is None:
            missing[filename] = names
            continue

        symbols = set(
            public_symbols(
                module
            )
        )

        found[filename] = [
            name
            for name in names
            if name in symbols
        ]

        absent = [
            name
            for name in names
            if name not in symbols
        ]

        if absent:
            missing[filename] = absent

    if missing:
        raise AssertionError(
            "Required public symbols missing: "
            + json.dumps(
                missing,
                indent=2,
            )
        )

    return {
        "required_modules": len(
            required
        ),
        "found": found,
    }


# ============================================================================
# 8. Architecture contract
# ============================================================================

def test_architecture_contract(
    runner: IntegrationTestRunner,
) -> Dict[str, Any]:
    module = runner.modules.get(
        "panini_api_server.py"
    )

    if module is None:
        raise AssertionError(
            "API module not loaded."
        )

    architecture = module.architecture()

    assert (
        architecture["name"]
        == "Panini Language Machine"
    )

    pipeline = architecture[
        "pipeline"
    ]

    assert len(
        pipeline
    ) >= 8

    component_names = [
        item.get(
            "component",
            "",
        )
        for item in pipeline
    ]

    expected_components = [
        "panini_core / panini_engine",
        "ashtadhyayi_compiler / scaled_panini_compiler",
        "karaka_dependency",
        "neuro_symbolic_panini",
        "paninian_english_llm",
        "neuro_symbolic_trainer",
        "paninian_vs_llm_benchmarker",
        "panini_api_server",
    ]

    missing = [
        item
        for item in expected_components
        if item not in component_names
    ]

    if missing:
        raise AssertionError(
            "Architecture contract missing: "
            + json.dumps(
                missing
            )
        )

    assert (
        "alpha * symbolic_bias"
        in architecture[
            "attention_contract"
        ]
    )

    return {
        "pipeline_stages": len(
            pipeline
        ),
        "attention_contract": architecture[
            "attention_contract"
        ],
    }


# ============================================================================
# 9. Benchmark integration
# ============================================================================

def test_benchmark_module(
    runner: IntegrationTestRunner,
) -> Dict[str, Any]:
    module = runner.modules.get(
        "paninian_vs_llm_benchmarker.py"
    )

    if module is None:
        raise AssertionError(
            "Benchmark module unavailable."
        )

    config = module.BenchmarkConfig(
        seed=11,
        epochs=3,
        learning_rate=0.08,
        symbolic_scale=1.0,
        samples=30,
        test_fraction=0.25,
        verbose=False,
    )

    report = module.run_benchmark(
        config
    )

    report_dict = json_safe(
        report
    )

    assert (
        "training_baseline"
        in report_dict
    )

    assert (
        "training_paninian"
        in report_dict
    )

    assert (
        "relative_improvement"
        in report_dict
    )

    return {
        "epochs": 3,
        "samples": 30,
        "benchmark_completed": True,
    }


def test_transformer_contract(
    runner: IntegrationTestRunner,
) -> Dict[str, Any]:
    module = runner.modules.get(
        "paninian_vs_llm_benchmarker.py"
    )

    if module is None:
        raise AssertionError(
            "Benchmark module unavailable."
        )

    spec = (
        module
        .transformer_integration_spec()
    )

    assert (
        "baseline"
        in spec
    )

    assert (
        "paninian"
        in spec
    )

    assert (
        "attention"
        in spec["paninian"]
    )

    assert (
        "symbolic_bias"
        in spec["paninian"][
            "attention"
        ]
    )

    assert (
        "language_model_loss"
        in spec["recommended_loss"]
    )

    return {
        "attention": spec[
            "paninian"
        ]["attention"],
        "loss": spec[
            "recommended_loss"
        ]["total"],
    }


# ============================================================================
# 10. API integration
# ============================================================================

def test_api_architecture(
    runner: IntegrationTestRunner,
) -> Dict[str, Any]:
    module = runner.modules.get(
        "panini_api_server.py"
    )

    if module is None:
        raise AssertionError(
            "API server module unavailable."
        )

    architecture = (
        module.architecture()
    )

    serialized = json.dumps(
        architecture,
        ensure_ascii=False,
    )

    assert (
        "Panini Language Machine"
        in serialized
    )

    return {
        "serializable": True,
        "bytes": len(
            serialized.encode(
                "utf-8"
            )
        ),
    }


def test_api_graph_normalization(
    runner: IntegrationTestRunner,
) -> Dict[str, Any]:
    module = runner.modules.get(
        "panini_api_server.py"
    )

    if module is None:
        raise AssertionError(
            "API server module unavailable."
        )

    analysis = {
        "stages": [
            {
                "stage": "synthetic",
                "status": "success",
                "data": {
                    "nodes": [
                        {
                            "id": "n1",
                            "label": "कर्ता",
                            "type": "Kartr",
                        },
                        {
                            "id": "n2",
                            "label": "फलम्",
                            "type": "Karma",
                        },
                    ],
                    "edges": [
                        {
                            "source": "n1",
                            "target": "n2",
                            "relation": "agent-object",
                        }
                    ],
                },
            }
        ]
    }

    graph = (
        module.graph_from_analysis(
            analysis
        )
    )

    assert (
        graph["node_count"] == 2
    )

    assert (
        graph["edge_count"] == 1
    )

    return graph


def test_api_input_validation(
    runner: IntegrationTestRunner,
) -> Dict[str, Any]:
    module = runner.modules.get(
        "panini_api_server.py"
    )

    if module is None:
        raise AssertionError(
            "API server module unavailable."
        )

    require_text = (
        module.require_text
    )

    try:
        require_text({})
        raise AssertionError(
            "Missing text did not fail."
        )
    except ValueError:
        pass

    try:
        require_text(
            {
                "text": ""
            }
        )
        raise AssertionError(
            "Empty text did not fail."
        )
    except ValueError:
        pass

    text = require_text(
        {
            "text": (
                "The merchant cuts the apple."
            )
        }
    )

    assert isinstance(
        text,
        str,
    )

    return {
        "validation": "pass",
    }


def test_api_server_construction(
    runner: IntegrationTestRunner,
) -> Dict[str, Any]:
    module = runner.modules.get(
        "panini_api_server.py"
    )

    if module is None:
        raise AssertionError(
            "API server module unavailable."
        )

    config = module.ServerConfig(
        host="127.0.0.1",
        port=0,
    )

    server = module.create_server(
        config
    )

    try:
        assert (
            server.panini_config.host
            == "127.0.0.1"
        )

        assert (
            server.panini_config.port
            == 0
        )

        return {
            "server_constructed": True,
        }

    finally:
        server.server_close()


# ============================================================================
# 11. Cross-layer tests
# ============================================================================

def test_benchmark_api_bridge(
    runner: IntegrationTestRunner,
) -> Dict[str, Any]:
    api = runner.modules.get(
        "panini_api_server.py"
    )

    benchmark = runner.modules.get(
        "paninian_vs_llm_benchmarker.py"
    )

    if api is None or benchmark is None:
        raise AssertionError(
            "Required modules unavailable."
        )

    payload = {
        "seed": 3,
        "epochs": 2,
        "samples": 20,
        "test_fraction": 0.25,
        "learning_rate": 0.08,
        "symbolic_scale": 1.0,
    }

    result = (
        api.run_benchmark_endpoint(
            payload
        )
    )

    result = json_safe(
        result
    )

    assert (
        "training_baseline"
        in result
    )

    assert (
        "training_paninian"
        in result
    )

    return {
        "bridge": "API -> benchmark",
        "completed": True,
    }


def test_analysis_export_contract(
    runner: IntegrationTestRunner,
) -> Dict[str, Any]:
    api = runner.modules.get(
        "panini_api_server.py"
    )

    if api is None:
        raise AssertionError(
            "API module unavailable."
        )

    payload = {
        "text": (
            "The merchant cuts the apple "
            "with a knife."
        ),
        "language": "english",
    }

    result = api.export_analysis(
        payload
    )

    assert (
        result["format"]
        == "panini-analysis-v1"
    )

    assert (
        "analysis"
        in result
    )

    assert (
        "graph"
        in result
    )

    assert (
        "architecture"
        in result
    )

    # Ensure it can cross a JSON API boundary.
    json.dumps(
        json_safe(result),
        ensure_ascii=False,
    )

    return {
        "format": result[
            "format"
        ],
        "json_serializable": True,
    }


# ============================================================================
# 12. Trainer integration
# ============================================================================

def test_trainer_module(
    runner: IntegrationTestRunner,
) -> Dict[str, Any]:
    module = runner.modules.get(
        "neuro_symbolic_trainer.py"
    )

    if module is None:
        raise AssertionError(
            "Trainer module unavailable."
        )

    symbols = set(
        public_symbols(
            module
        )
    )

    # The trainer may expose one of several names depending on the earlier
    # implementation. We only require that it expose at least one executable
    # training entry point or configuration object.
    candidates = [
        "Trainer",
        "NeuroSymbolicTrainer",
        "train",
        "train_model",
        "TrainingConfig",
    ]

    found = [
        name
        for name in candidates
        if name in symbols
    ]

    if not found:
        raise AssertionError(
            "No recognized trainer API found."
        )

    return {
        "recognized_symbols": found,
    }


# ============================================================================
# 13. Parser / compiler discovery
# ============================================================================

def test_linguistic_layers_present(
    runner: IntegrationTestRunner,
) -> Dict[str, Any]:
    expected = [
        "panini_core.py",
        "panini_engine.py",
        "ashtadhyayi_compiler.py",
        "scaled_panini_compiler.py",
        "marathi_parser.py",
        "karaka_dependency.py",
    ]

    loaded = [
        filename
        for filename in expected
        if filename in runner.modules
    ]

    if len(loaded) != len(
        expected
    ):
        raise AssertionError(
            "Not all linguistic layers loaded: "
            + json.dumps(
                loaded
            )
        )

    symbol_counts = {
        filename: len(
            public_symbols(
                runner.modules[
                    filename
                ]
            )
        )
        for filename in expected
    }

    return {
        "layers": loaded,
        "public_symbol_counts": symbol_counts,
    }


# ============================================================================
# 14. Neuro-symbolic chain
# ============================================================================

def test_neuro_symbolic_layers_present(
    runner: IntegrationTestRunner,
) -> Dict[str, Any]:
    expected = [
        "neuro_symbolic_panini.py",
        "paninian_english_llm.py",
        "neuro_symbolic_trainer.py",
        "paninian_vs_llm_benchmarker.py",
    ]

    missing = [
        filename
        for filename in expected
        if filename not in runner.modules
    ]

    if missing:
        raise AssertionError(
            "Missing neuro-symbolic layers: "
            + json.dumps(
                missing
            )
        )

    return {
        "layers": expected,
        "complete": True,
    }


# ============================================================================
# 15. End-to-end health
# ============================================================================

def test_end_to_end_health(
    runner: IntegrationTestRunner,
) -> Dict[str, Any]:
    api = runner.modules.get(
        "panini_api_server.py"
    )

    if api is None:
        raise AssertionError(
            "API layer unavailable."
        )

    # Architecture.
    architecture = (
        api.architecture()
    )

    assert architecture[
        "status"
    ] == (
        "experimental_research_framework"
    )

    # Input validation.
    text = api.require_text(
        {
            "text": (
                "The student reads the book."
            )
        }
    )

    assert text

    # Analysis should produce a valid response even when optional
    # project-specific adapters are unavailable.
    analysis = api.analyze_text(
        text,
        language="english",
    )

    assert (
        analysis["text"]
        == text
    )

    assert (
        analysis["status"]
        in {
            "success",
            "partial",
        }
    )

    assert isinstance(
        analysis["stages"],
        list,
    )

    assert isinstance(
        analysis["errors"],
        list,
    )

    # Graph boundary.
    graph = (
        api.graph_from_analysis(
            analysis
        )
    )

    assert isinstance(
        graph["nodes"],
        list,
    )

    assert isinstance(
        graph["edges"],
        list,
    )

    # JSON boundary.
    json.dumps(
        json_safe(
            {
                "analysis": analysis,
                "graph": graph,
            }
        ),
        ensure_ascii=False,
    )

    return {
        "analysis_status": analysis[
            "status"
        ],
        "stage_count": len(
            analysis[
                "stages"
            ]
        ),
        "node_count": graph[
            "node_count"
        ],
        "edge_count": graph[
            "edge_count"
        ],
        "json_boundary": True,
    }


# ============================================================================
# 16. Full suite
# ============================================================================

def run_integration_suite(
    project_dir: Path = PROJECT_DIR,
    verbose: bool = False,
) -> IntegrationReport:
    started_at = time.time()

    runner = IntegrationTestRunner(
        project_dir=project_dir,
        verbose=verbose,
    )

    tests = [
        (
            "all_files_exist",
            "filesystem",
            test_all_files_exist,
        ),
        (
            "python_syntax",
            "syntax",
            test_python_syntax,
        ),
        (
            "modules_import",
            "imports",
            test_modules_import,
        ),
        (
            "core_symbols",
            "api_contract",
            test_core_symbols,
        ),
        (
            "architecture_contract",
            "architecture",
            test_architecture_contract,
        ),
        (
            "linguistic_layers_present",
            "linguistics",
            test_linguistic_layers_present,
        ),
        (
            "neuro_symbolic_layers_present",
            "neuro_symbolic",
            test_neuro_symbolic_layers_present,
        ),
        (
            "benchmark_module",
            "benchmark",
            test_benchmark_module,
        ),
        (
            "transformer_contract",
            "benchmark",
            test_transformer_contract,
        ),
        (
            "api_architecture",
            "api",
            test_api_architecture,
        ),
        (
            "api_graph_normalization",
            "api",
            test_api_graph_normalization,
        ),
        (
            "api_input_validation",
            "api",
            test_api_input_validation,
        ),
        (
            "api_server_construction",
            "api",
            test_api_server_construction,
        ),
        (
            "benchmark_api_bridge",
            "cross_layer",
            test_benchmark_api_bridge,
        ),
        (
            "analysis_export_contract",
            "cross_layer",
            test_analysis_export_contract,
        ),
        (
            "trainer_module",
            "training",
            test_trainer_module,
        ),
        (
            "end_to_end_health",
            "end_to_end",
            test_end_to_end_health,
        ),
    ]

    for name, category, function in tests:
        runner.run(
            name,
            category,
            lambda fn=function: fn(
                runner
            ),
        )

    finished_at = time.time()

    status = (
        "SYSTEM_INTEGRATION_PASS"
        if runner.failed == 0
        else "SYSTEM_INTEGRATION_FAIL"
    )

    return IntegrationReport(
        project_dir=str(
            runner.project_dir
        ),
        started_at=started_at,
        finished_at=finished_at,
        passed=runner.passed,
        failed=runner.failed,
        skipped=runner.skipped,
        results=runner.results,
        system_status=status,
    )


# ============================================================================
# 17. Report rendering
# ============================================================================

def render_report(
    report: IntegrationReport,
) -> str:
    lines = [
        "=" * 108,
        "PANINI LANGUAGE MACHINE — SYSTEM INTEGRATION TEST",
        "=" * 108,
        "",
        f"Project : {report.project_dir}",
        f"Passed  : {report.passed}",
        f"Failed  : {report.failed}",
        f"Skipped : {report.skipped}",
        f"Time    : "
        f"{report.finished_at - report.started_at:.3f}s",
        "",
        "-" * 108,
        f"{'STATUS':<8}"
        f"{'CATEGORY':<18}"
        f"{'TEST':<42}"
        f"{'TIME':>12}",
        "-" * 108,
    ]

    for result in report.results:
        lines.append(
            f"{result.status:<8}"
            f"{result.category:<18}"
            f"{result.name:<42}"
            f"{result.duration_seconds:>12.4f}"
        )

        if result.message:
            lines.append(
                f"         {result.message}"
            )

    lines.extend([
        "",
        "=" * 108,
        report.system_status,
        "=" * 108,
    ])

    if report.failed:
        lines.extend([
            "",
            "FAILED TESTS",
            "-" * 108,
        ])

        for result in report.results:
            if result.status == "FAIL":
                lines.append(
                    f"* {result.category}/"
                    f"{result.name}: "
                    f"{result.message}"
                )

    return "\n".join(
        lines
    )


# ============================================================================
# 18. JSON export
# ============================================================================

def write_report(
    report: IntegrationReport,
    path: str | Path,
) -> Path:
    destination = Path(
        path
    )

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
# 19. CLI
# ============================================================================

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="integration_tests",
        description=(
            "Run end-to-end integration tests for "
            "the 14-file Panini Language Machine."
        ),
    )

    parser.add_argument(
        "--project-dir",
        default=str(
            PROJECT_DIR
        ),
        help=(
            "Directory containing the 14 project files."
        ),
    )

    parser.add_argument(
        "--json",
        metavar="PATH",
        help=(
            "Write detailed integration report to JSON."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print each test as it runs."
        ),
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "Reserved for future CI execution."
        ),
    )

    return parser


def main(
    argv: Optional[Sequence[str]] = None,
) -> int:
    parser = build_cli()

    args = parser.parse_args(
        argv
    )

    project_dir = Path(
        args.project_dir
    ).resolve()

    report = run_integration_suite(
        project_dir=project_dir,
        verbose=args.verbose,
    )

    print(
        render_report(
            report
        )
    )

    if args.json:
        destination = write_report(
            report,
            args.json,
        )

        print(
            f"\nReport written to: {destination}"
        )

    return (
        0
        if report.failed == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
