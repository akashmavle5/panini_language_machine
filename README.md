# Akash's Panini Language Machine — UI v2

A React / Next.js / Node.js user interface built around Material UI's free dashboard-template approach.

Material UI publishes free React templates including a dashboard and CRUD dashboard; this project follows that free dashboard pattern: responsive permanent/temporary drawer, app bar, cards, tabs, metrics, forms, status states and a research-workspace layout. See:
https://mui.com/material-ui/getting-started/templates/

## What is included

All 14 Python files are represented in the UI:

01 panini_core.py
02 panini_engine.py
03 ashtadhyayi_compiler.py
04 scaled_panini_compiler.py
05 marathi_parser.py
06 marathi_interactive_tool.py
07 panini_exporter.py
08 karaka_dependency.py
09 neuro_symbolic_panini.py
10 paninian_english_llm.py
11 neuro_symbolic_trainer.py
12 paninian_vs_llm_benchmarker.py
13 panini_api_server.py
14 integration_tests.py

Every module has:
- an Open workspace action
- a Call action
- module metadata
- an appropriate UI action mapping
- JSON output/status area

## Architecture

Browser
  -> Next.js / React
  -> Material UI
  -> Node.js gateway :3001
  -> Python panini_api_server.py :8787
  -> 14-file Panini system

## Start

From this UI directory:

```bash
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

In another terminal, from the Python project directory:

```bash
python panini_api_server.py
```

Expected Python API:

```text
http://127.0.0.1:8787
```

If the Python API uses another address:

PowerShell:

```powershell
$env:PYTHON_API="http://127.0.0.1:9000"
npm run gateway
```

## Production

```bash
npm run build
npm start
```

## Important implementation note

The UI does not invent the internals of the 14 Python programs. It uses the API exposed by `panini_api_server.py`. For module-level calls, the Node gateway first attempts `/module/<file>` and falls back to a registry response if the current Python API does not yet expose that route.

This makes the UI runnable against the current File 13 implementation while clearly identifying which module is being addressed.

## Error fixes in v2

- Added all missing Material UI icon imports.
- Removed fragile/undefined icon references.
- Centralized module definitions in `src/config.js`.
- Added a dedicated Node gateway route for all 14 modules.
- Added loading/error states.
- Added responsive MUI drawer/navigation.
- Added module-specific workspaces.
- Added API/architecture/registry views.
- Added benchmark and export actions.
- Added integration status view.
- Kept Next.js 15.5.23 and React 19 compatible.
