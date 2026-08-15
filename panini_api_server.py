"""
panini_api_server.py
====================

Panini Language Machine — API Server

File 13/14.

Purpose
-------
Expose the complete Panini computational pipeline through a small HTTP API.

Pipeline
--------
    Text
      ↓
    Paninian analysis
      ↓
    Kāraka / dependency graph
      ↓
    Neuro-symbolic representation
      ↓
    Benchmark / training interfaces

Design principles
-----------------
* Python standard library for the server itself.
* Existing project files are treated as the source of the computational
  components; this file does not silently replace them.
* Imports are optional and fail gracefully so the server can still expose
  health, architecture and metadata endpoints when an earlier module is
  unavailable.
* JSON is the primary interchange format.
* The API is deliberately small and can later be mounted behind FastAPI,
  Django, Flask, or a production gateway without changing the core contracts.

Endpoints
---------
GET
    /health
    /architecture
    /modules
    /benchmark/spec

POST
    /analyze
    /graph
    /benchmark
    /export

The server is intended as the integration boundary for File 14.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse


# ============================================================================
# 1. Project configuration
# ============================================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODULE_FILES = [
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
]


# ============================================================================
# 2. Generic serialization
# ============================================================================

def json_safe(value: Any) -> Any:
    """
    Convert project objects into JSON-safe values without changing them.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return json_safe(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            json_safe(item)
            for item in value
        ]

    if hasattr(value, "__dict__"):
        return {
            str(key): json_safe(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }

    return str(value)


# ============================================================================
# 3. Dynamic module loader
# ============================================================================

class ModuleRegistry:
    """
    Load the project's earlier Python files without making the API server
    dependent on one particular import/package layout.
    """

    def __init__(
        self,
        project_dir: Path = PROJECT_DIR,
    ) -> None:
        self.project_dir = Path(project_dir)
        self.modules: Dict[str, Any] = {}
        self.errors: Dict[str, str] = {}

    def load(
        self,
        filename: str,
    ) -> Optional[Any]:
        if filename in self.modules:
            return self.modules[filename]

        path = self.project_dir / filename

        if not path.exists():
            self.errors[filename] = (
                f"Module file not found: {path}"
            )
            return None

        module_name = (
            "panini_project_"
            + path.stem.replace("-", "_")
        )

        try:
            spec = importlib.util.spec_from_file_location(
                module_name,
                str(path),
            )

            if spec is None or spec.loader is None:
                raise ImportError(
                    f"Unable to create import specification for {path}"
                )

            module = importlib.util.module_from_spec(
                spec
            )

            # Register dynamically loaded modules before execution.
            # This is required by dataclasses and other introspective
            # libraries that resolve cls.__module__ through sys.modules.
            sys.modules[module_name] = module

            # Make sibling imports work for project modules.
            if str(self.project_dir) not in sys.path:
                sys.path.insert(
                    0,
                    str(self.project_dir),
                )

            spec.loader.exec_module(module)

            self.modules[filename] = module
            return module

        except Exception as exc:
            self.errors[filename] = (
                f"{type(exc).__name__}: {exc}"
            )
            return None

    def load_all(self) -> Dict[str, Any]:
        for filename in MODULE_FILES:
            self.load(filename)

        return dict(self.modules)

    def status(self) -> Dict[str, Any]:
        loaded = []
        missing = []
        failed = []

        for filename in MODULE_FILES:
            path = self.project_dir / filename

            if filename in self.modules:
                loaded.append(filename)
            elif not path.exists():
                missing.append(filename)
            else:
                failed.append(filename)

        return {
            "project_dir": str(self.project_dir),
            "expected_modules": MODULE_FILES,
            "loaded": loaded,
            "missing": missing,
            "failed": failed,
            "errors": dict(self.errors),
        }


REGISTRY = ModuleRegistry()


# ============================================================================
# 4. Architecture contract
# ============================================================================

def architecture() -> Dict[str, Any]:
    return {
        "name": "Panini Language Machine",
        "version": "0.1.0-experimental",
        "pipeline": [
            {
                "stage": 1,
                "name": "Text input",
                "component": "linguistic sentence",
            },
            {
                "stage": 2,
                "name": "Paninian analysis",
                "component": "panini_core / panini_engine",
            },
            {
                "stage": 3,
                "name": "Grammar compilation",
                "component": (
                    "ashtadhyayi_compiler / "
                    "scaled_panini_compiler"
                ),
            },
            {
                "stage": 4,
                "name": "Kāraka dependency",
                "component": "karaka_dependency",
            },
            {
                "stage": 5,
                "name": "Neuro-symbolic representation",
                "component": "neuro_symbolic_panini",
            },
            {
                "stage": 6,
                "name": "English structural automata",
                "component": "paninian_english_llm",
            },
            {
                "stage": 7,
                "name": "Neuro-symbolic training",
                "component": "neuro_symbolic_trainer",
            },
            {
                "stage": 8,
                "name": "Benchmarking",
                "component": "paninian_vs_llm_benchmarker",
            },
            {
                "stage": 9,
                "name": "API integration",
                "component": "panini_api_server",
            },
        ],
        "attention_contract": (
            "QK^T / sqrt(d) + alpha * symbolic_bias"
        ),
        "training_contract": (
            "L_total = L_language_model + "
            "lambda * L_symbolic_consistency"
        ),
        "status": "experimental_research_framework",
    }


# ============================================================================
# 5. Input validation
# ============================================================================

def require_mapping(
    payload: Any,
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            "Request body must be a JSON object."
        )

    return payload


def require_text(
    payload: Mapping[str, Any],
) -> str:
    text = payload.get("text")

    if not isinstance(text, str):
        raise ValueError(
            'Request field "text" must be a string.'
        )

    text = text.strip()

    if not text:
        raise ValueError(
            'Request field "text" cannot be empty.'
        )

    if len(text) > 20000:
        raise ValueError(
            "Text exceeds the 20,000 character limit."
        )

    return text


# ============================================================================
# 6. Module-specific adapters
# ============================================================================

def module_summary() -> Dict[str, Any]:
    """
    Return import status and public symbols.

    This is intentionally diagnostic rather than prescriptive: it shows
    what the existing files actually expose.
    """

    REGISTRY.load_all()

    result: Dict[str, Any] = {}

    for filename in MODULE_FILES:
        module = REGISTRY.modules.get(
            filename
        )

        if module is None:
            result[filename] = {
                "loaded": False,
                "error": REGISTRY.errors.get(
                    filename
                ),
            }
            continue

        public_symbols = sorted(
            name
            for name in dir(module)
            if not name.startswith("_")
        )

        result[filename] = {
            "loaded": True,
            "public_symbols": public_symbols,
        }

    return result


def find_callable(
    module: Any,
    candidates: Sequence[str],
) -> Optional[Any]:
    if module is None:
        return None

    for name in candidates:
        candidate = getattr(
            module,
            name,
            None,
        )

        if callable(candidate):
            return candidate

    return None


def instantiate_candidates(
    module: Any,
    candidates: Sequence[str],
) -> Optional[Any]:
    if module is None:
        return None

    for name in candidates:
        cls = getattr(
            module,
            name,
            None,
        )

        if not isinstance(
            cls,
            type,
        ):
            continue

        # Prefer no-argument constructors.
        try:
            return cls()
        except TypeError:
            continue
        except Exception:
            continue

    return None


# ============================================================================
# 7. Analysis pipeline
# ============================================================================

@dataclass
class AnalysisResult:
    text: str
    language: str
    status: str
    stages: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


def analyze_text(
    text: str,
    language: str = "auto",
) -> Dict[str, Any]:
    """
    Run the strongest available project-specific analysis.

    Because the earlier files may have different public APIs, the adapter
    discovers their callable interfaces rather than inventing one.

    If a module cannot be used, the response records the reason explicitly.
    """

    REGISTRY.load_all()

    stages: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    # ----------------------------------------------------------------------
    # Stage A: basic input
    # ----------------------------------------------------------------------

    stages.append({
        "stage": "input",
        "status": "success",
        "data": {
            "text": text,
            "language": language,
            "character_count": len(text),
        },
    })

    # ----------------------------------------------------------------------
    # Stage B: English VGA if requested/appropriate
    # ----------------------------------------------------------------------

    english_module = REGISTRY.modules.get(
        "paninian_english_llm.py"
    )

    english_result = None

    if (
        language.lower()
        in {"en", "english", "auto"}
    ):
        if english_module is not None:
            cls = getattr(
                english_module,
                "VyakaranaGraphAutomata",
                None,
            )

            if cls is not None:
                try:
                    engine = cls()

                    # Known interface used by File 10.
                    for method_name in (
                        "analyze",
                        "parse",
                        "process",
                    ):
                        method = getattr(
                            engine,
                            method_name,
                            None,
                        )

                        if callable(method):
                            try:
                                english_result = method(
                                    text
                                )
                                break
                            except TypeError:
                                continue

                    if english_result is not None:
                        stages.append({
                            "stage": "english_vga",
                            "status": "success",
                            "data": json_safe(
                                english_result
                            ),
                        })
                    else:
                        stages.append({
                            "stage": "english_vga",
                            "status": "available_but_no_compatible_method",
                            "data": {
                                "class": (
                                    "VyakaranaGraphAutomata"
                                ),
                            },
                        })

                except Exception as exc:
                    errors.append({
                        "stage": "english_vga",
                        "error": str(exc),
                        "type": type(exc).__name__,
                    })
            else:
                stages.append({
                    "stage": "english_vga",
                    "status": "class_not_found",
                })
        else:
            stages.append({
                "stage": "english_vga",
                "status": "module_unavailable",
            })

    # ----------------------------------------------------------------------
    # Stage C: Kāraka dependency adapter
    # ----------------------------------------------------------------------

    karaka_module = REGISTRY.modules.get(
        "karaka_dependency.py"
    )

    karaka_result = None

    if karaka_module is not None:
        analyzer = instantiate_candidates(
            karaka_module,
            [
                "KarakaDependencyAnalyzer",
                "KarakaAnalyzer",
                "DependencyAnalyzer",
            ],
        )

        if analyzer is not None:
            for method_name in (
                "analyze",
                "parse",
                "build_graph",
                "process",
            ):
                method = getattr(
                    analyzer,
                    method_name,
                    None,
                )

                if callable(method):
                    try:
                        karaka_result = method(
                            text
                        )
                        break
                    except TypeError:
                        continue

            if karaka_result is not None:
                stages.append({
                    "stage": "karaka_dependency",
                    "status": "success",
                    "data": json_safe(
                        karaka_result
                    ),
                })
            else:
                stages.append({
                    "stage": "karaka_dependency",
                    "status": (
                        "available_but_no_compatible_method"
                    ),
                })
        else:
            stages.append({
                "stage": "karaka_dependency",
                "status": "class_not_found",
            })

    # ----------------------------------------------------------------------
    # Stage D: neuro-symbolic adapter
    # ----------------------------------------------------------------------

    neuro_module = REGISTRY.modules.get(
        "neuro_symbolic_panini.py"
    )

    neuro_result = None

    if neuro_module is not None:
        callable_adapter = find_callable(
            neuro_module,
            [
                "analyze",
                "parse",
                "process",
                "build_graph",
                "build_representation",
            ],
        )

        if callable_adapter is not None:
            try:
                neuro_result = callable_adapter(
                    text
                )

                stages.append({
                    "stage": "neuro_symbolic",
                    "status": "success",
                    "data": json_safe(
                        neuro_result
                    ),
                })
            except Exception as exc:
                errors.append({
                    "stage": "neuro_symbolic",
                    "error": str(exc),
                    "type": type(exc).__name__,
                })
        else:
            stages.append({
                "stage": "neuro_symbolic",
                "status": (
                    "module_loaded_but_no_known_callable"
                ),
            })

    # ----------------------------------------------------------------------
    # Final status
    # ----------------------------------------------------------------------

    status = (
        "success"
        if not errors
        else "partial"
    )

    result = AnalysisResult(
        text=text,
        language=language,
        status=status,
        stages=stages,
        errors=errors,
    )

    return result.to_dict()


# ============================================================================
# 8. Graph endpoint
# ============================================================================

def graph_from_analysis(
    analysis: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Normalize available analysis information into a graph-shaped contract.

    Nodes and edges are only emitted when earlier components provide enough
    information to support them.
    """

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    seen_nodes = set()

    def add_node(
        node_id: str,
        label: str,
        node_type: str,
        **extra: Any,
    ) -> None:
        if node_id in seen_nodes:
            return

        seen_nodes.add(node_id)

        node = {
            "id": node_id,
            "label": label,
            "type": node_type,
        }

        node.update(extra)
        nodes.append(node)

    for stage in analysis.get(
        "stages",
        [],
    ):
        data = stage.get(
            "data"
        )

        if not isinstance(
            data,
            dict,
        ):
            continue

        # Common node-shaped keys.
        for key in (
            "nodes",
            "entities",
            "tokens",
        ):
            values = data.get(key)

            if not isinstance(
                values,
                list,
            ):
                continue

            for index, value in enumerate(
                values
            ):
                if isinstance(
                    value,
                    dict,
                ):
                    node_id = str(
                        value.get(
                            "id",
                            value.get(
                                "token_id",
                                index,
                            ),
                        )
                    )

                    label = str(
                        value.get(
                            "label",
                            value.get(
                                "text",
                                node_id,
                            ),
                        )
                    )

                    node_type = str(
                        value.get(
                            "type",
                            "linguistic",
                        )
                    )

                    add_node(
                        node_id,
                        label,
                        node_type,
                        **{
                            key: json_safe(val)
                            for key, val
                            in value.items()
                            if key
                            not in {
                                "id",
                                "label",
                                "text",
                                "type",
                            }
                        },
                    )
                else:
                    node_id = f"node_{index}"
                    add_node(
                        node_id,
                        str(value),
                        "linguistic",
                    )

        # Common edge-shaped keys.
        for key in (
            "edges",
            "relations",
            "dependencies",
        ):
            values = data.get(key)

            if not isinstance(
                values,
                list,
            ):
                continue

            for value in values:
                if isinstance(
                    value,
                    dict,
                ):
                    source = value.get(
                        "source",
                        value.get(
                            "from"
                        ),
                    )

                    target = value.get(
                        "target",
                        value.get(
                            "to"
                        ),
                    )

                    relation = value.get(
                        "relation",
                        value.get(
                            "label",
                            value.get(
                                "type",
                                "relation",
                            ),
                        ),
                    )

                    if (
                        source is not None
                        and target is not None
                    ):
                        edges.append({
                            "source": str(
                                source
                            ),
                            "target": str(
                                target
                            ),
                            "relation": str(
                                relation
                            ),
                        })

                elif isinstance(
                    value,
                    (list, tuple),
                ) and len(value) >= 2:
                    edges.append({
                        "source": str(
                            value[0]
                        ),
                        "target": str(
                            value[1]
                        ),
                        "relation": (
                            str(value[2])
                            if len(value) >= 3
                            else "relation"
                        ),
                    })

    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


# ============================================================================
# 9. Benchmark endpoint
# ============================================================================

def run_benchmark_endpoint(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    benchmark_module = REGISTRY.load(
        "paninian_vs_llm_benchmarker.py"
    )

    if benchmark_module is None:
        raise RuntimeError(
            "paninian_vs_llm_benchmarker.py "
            "could not be loaded."
        )

    config_cls = getattr(
        benchmark_module,
        "BenchmarkConfig",
        None,
    )

    run_benchmark = getattr(
        benchmark_module,
        "run_benchmark",
        None,
    )

    if not callable(
        run_benchmark
    ):
        raise RuntimeError(
            "run_benchmark() is not exposed by "
            "paninian_vs_llm_benchmarker.py."
        )

    allowed = {
        "seed",
        "epochs",
        "learning_rate",
        "symbolic_scale",
        "hidden_size",
        "samples",
        "test_fraction",
        "verbose",
    }

    options = {
        key: value
        for key, value in payload.items()
        if key in allowed
    }

    if config_cls is not None:
        config = config_cls(
            **options
        )
        result = run_benchmark(
            config
        )
    else:
        result = run_benchmark()

    return json_safe(
        result
    )


# ============================================================================
# 10. Export endpoint
# ============================================================================

def export_analysis(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    text = require_text(
        payload
    )

    language = str(
        payload.get(
            "language",
            "auto",
        )
    )

    analysis = analyze_text(
        text,
        language=language,
    )

    graph = graph_from_analysis(
        analysis
    )

    return {
        "format": "panini-analysis-v1",
        "analysis": analysis,
        "graph": graph,
        "architecture": architecture(),
    }


# ============================================================================
# 11. HTTP server
# ============================================================================

@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8787
    max_body_bytes: int = 2_000_000


class PaniniRequestHandler(
    BaseHTTPRequestHandler
):
    server_version = (
        "PaniniLanguageMachine/0.1"
    )

    def _send_json(
        self,
        payload: Any,
        status: int = 200,
    ) -> None:
        body = json.dumps(
            json_safe(payload),
            ensure_ascii=False,
            indent=2,
        ).encode(
            "utf-8"
        )

        self.send_response(
            status
        )

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )

        self.end_headers()
        self.wfile.write(
            body
        )

    def _read_json(self) -> Dict[str, Any]:
        raw_length = self.headers.get(
            "Content-Length",
            "0",
        )

        try:
            length = int(
                raw_length
            )
        except ValueError:
            raise ValueError(
                "Invalid Content-Length header."
            )

        max_body = getattr(
            self.server,
            "panini_config",
            ServerConfig(),
        ).max_body_bytes

        if length > max_body:
            raise ValueError(
                f"Request body exceeds {max_body} bytes."
            )

        body = self.rfile.read(
            length
        )

        if not body:
            return {}

        try:
            payload = json.loads(
                body.decode(
                    "utf-8"
                )
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON: {exc}"
            )

        return require_mapping(
            payload
        )

    def do_OPTIONS(self) -> None:
        self.send_response(
            204
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS",
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type",
        )

        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(
            self.path
        )

        route = parsed.path.rstrip(
            "/"
        ) or "/"

        try:
            if route == "/health":
                self._send_json(
                    {
                        "status": "ok",
                        "service": (
                            "Panini Language Machine"
                        ),
                        "version": "0.1.0",
                        "modules": REGISTRY.status(),
                    }
                )
                return

            if route == "/architecture":
                self._send_json(
                    architecture()
                )
                return

            if route == "/modules":
                self._send_json(
                    module_summary()
                )
                return

            if route == "/benchmark/spec":
                benchmark_module = REGISTRY.load(
                    "paninian_vs_llm_benchmarker.py"
                )

                if benchmark_module is not None:
                    spec_fn = getattr(
                        benchmark_module,
                        "transformer_integration_spec",
                        None,
                    )

                    if callable(
                        spec_fn
                    ):
                        self._send_json(
                            spec_fn()
                        )
                        return

                self._send_json(
                    {
                        "status": (
                            "benchmark module unavailable"
                        )
                    },
                    status=503,
                )
                return

            self._send_json(
                {
                    "error": "Not found",
                    "path": route,
                    "available_routes": [
                        "GET /health",
                        "GET /architecture",
                        "GET /modules",
                        "GET /benchmark/spec",
                        "POST /analyze",
                        "POST /graph",
                        "POST /benchmark",
                        "POST /export",
                    ],
                },
                status=404,
            )

        except Exception as exc:
            self._send_json(
                {
                    "error": str(exc),
                    "type": type(exc).__name__,
                },
                status=500,
            )

    def do_POST(self) -> None:
        parsed = urlparse(
            self.path
        )

        route = parsed.path.rstrip(
            "/"
        ) or "/"

        try:
            payload = self._read_json()

            if route == "/analyze":
                text = require_text(
                    payload
                )

                language = str(
                    payload.get(
                        "language",
                        "auto",
                    )
                )

                result = analyze_text(
                    text,
                    language=language,
                )

                self._send_json(
                    result
                )
                return

            if route == "/graph":
                text = require_text(
                    payload
                )

                language = str(
                    payload.get(
                        "language",
                        "auto",
                    )
                )

                analysis = analyze_text(
                    text,
                    language=language,
                )

                self._send_json(
                    graph_from_analysis(
                        analysis
                    )
                )
                return

            if route == "/benchmark":
                result = run_benchmark_endpoint(
                    payload
                )

                self._send_json(
                    result
                )
                return

            if route == "/export":
                result = export_analysis(
                    payload
                )

                self._send_json(
                    result
                )
                return

            self._send_json(
                {
                    "error": "Not found",
                    "path": route,
                },
                status=404,
            )

        except ValueError as exc:
            self._send_json(
                {
                    "error": str(exc),
                    "type": "ValueError",
                },
                status=400,
            )

        except Exception as exc:
            self._send_json(
                {
                    "error": str(exc),
                    "type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )

    def log_message(
        self,
        format: str,
        *args: Any,
    ) -> None:
        # Keep the server console clean while retaining useful access logs.
        print(
            f"[PaniniAPI] {self.address_string()} "
            f"{format % args}"
        )


# ============================================================================
# 12. Server lifecycle
# ============================================================================

def create_server(
    config: Optional[ServerConfig] = None,
) -> ThreadingHTTPServer:
    config = config or ServerConfig()

    server = ThreadingHTTPServer(
        (
            config.host,
            config.port,
        ),
        PaniniRequestHandler,
    )

    server.panini_config = config

    return server


def serve(
    config: Optional[ServerConfig] = None,
) -> None:
    config = config or ServerConfig()

    # Load modules before opening the port so startup diagnostics are visible.
    REGISTRY.load_all()

    server = create_server(
        config
    )

    print(
        "Panini Language Machine API"
    )
    print(
        f"Listening on http://{config.host}:{config.port}"
    )
    print(
        "Endpoints:"
    )
    print(
        "  GET  /health"
    )
    print(
        "  GET  /architecture"
    )
    print(
        "  GET  /modules"
    )
    print(
        "  GET  /benchmark/spec"
    )
    print(
        "  POST /analyze"
    )
    print(
        "  POST /graph"
    )
    print(
        "  POST /benchmark"
    )
    print(
        "  POST /export"
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(
            "\nStopping Panini API..."
        )
    finally:
        server.server_close()


# ============================================================================
# 13. Local endpoint smoke tests
# ============================================================================

def self_test() -> None:
    """
    Offline self-test.

    This validates:
      * syntax-level server construction
      * JSON serialization
      * architecture contract
      * module registry
      * input validation
      * graph normalization
      * benchmark adapter when File 12 is available

    It does not bind a network port.
    """

    config = ServerConfig(
        host="127.0.0.1",
        port=0,
    )

    assert config.port == 0

    arch = architecture()

    assert (
        arch["name"]
        == "Panini Language Machine"
    )

    assert len(
        arch["pipeline"]
    ) >= 8

    assert (
        "alpha * symbolic_bias"
        in arch["attention_contract"]
    )

    try:
        require_text({})
        raise AssertionError(
            "Missing text should fail."
        )
    except ValueError:
        pass

    assert (
        require_text(
            {
                "text": "Rama gives a book."
            }
        )
        == "Rama gives a book."
    )

    fake_analysis = {
        "stages": [
            {
                "stage": "test",
                "status": "success",
                "data": {
                    "nodes": [
                        {
                            "id": "merchant",
                            "label": "merchant",
                            "type": "Kartā",
                        },
                        {
                            "id": "apple",
                            "label": "apple",
                            "type": "Karma",
                        },
                    ],
                    "edges": [
                        {
                            "source": "merchant",
                            "target": "apple",
                            "relation": "semantic",
                        }
                    ],
                },
            }
        ]
    }

    graph = graph_from_analysis(
        fake_analysis
    )

    assert graph["node_count"] == 2
    assert graph["edge_count"] == 1

    REGISTRY.load_all()

    # File 12 should be discoverable in the current project directory.
    benchmark_path = (
        PROJECT_DIR
        / "paninian_vs_llm_benchmarker.py"
    )

    if benchmark_path.exists():
        benchmark = REGISTRY.load(
            "paninian_vs_llm_benchmarker.py"
        )

        assert benchmark is not None

        spec_fn = getattr(
            benchmark,
            "transformer_integration_spec",
            None,
        )

        assert callable(
            spec_fn
        )

        spec = spec_fn()

        assert (
            "paninian"
            in spec
        )

    serialized = json.dumps(
        json_safe(
            {
                "architecture": arch,
                "graph": graph,
            }
        )
    )

    assert (
        "Panini Language Machine"
        in serialized
    )

    # Server can be constructed without binding a public port.
    server = create_server(
        config
    )

    assert (
        server.panini_config.host
        == "127.0.0.1"
    )

    server.server_close()


# ============================================================================
# 14. CLI
# ============================================================================

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="panini_api_server",
        description=(
            "Serve the Panini Language Machine through a JSON HTTP API."
        ),
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8787,
        help="TCP port.",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run offline self-tests.",
    )

    parser.add_argument(
        "--modules",
        action="store_true",
        help="Print module status and exit.",
    )

    parser.add_argument(
        "--architecture",
        action="store_true",
        help="Print architecture and exit.",
    )

    parser.add_argument(
        "--analyze",
        metavar="TEXT",
        help="Analyze text locally without starting the server.",
    )

    parser.add_argument(
        "--language",
        default="auto",
        help="Language hint for --analyze.",
    )

    return parser


def main(
    argv: Optional[Sequence[str]] = None,
) -> int:
    parser = build_cli()
    args = parser.parse_args(
        argv
    )

    if args.self_test:
        self_test()
        print(
            "SELF_TEST_PASS"
        )
        return 0

    if args.modules:
        print(
            json.dumps(
                module_summary(),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.architecture:
        print(
            json.dumps(
                architecture(),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.analyze:
        result = analyze_text(
            args.analyze,
            language=args.language,
        )

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

        return 0

    if not (
        1 <= args.port <= 65535
    ):
        parser.error(
            "--port must be between 1 and 65535."
        )

    serve(
        ServerConfig(
            host=args.host,
            port=args.port,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
