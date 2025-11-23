#!/usr/bin/env python3
"""
Run the local LangGraph graph (Essay Writer, Ghostwriter, or Deep Researcher) programmatically in Python.

Examples:
  # Non-stream, inline message; write final output to file
  uv run python scripts/run_graph.py --graph "Essay Writer" --message "AI in education" --output out/essay.md

  # Ghostwriter with style (non-stream)
  uv run python scripts/run_graph.py --graph "Ghostwriter" --message "Write about X" --output out/doc.md

  # Read prompt from file
  uv run python scripts/run_graph.py --graph "Essay Writer" --input-file prompt.txt --output out/essay.md

  # Stream node events to console
  uv run python scripts/run_graph.py --graph "Essay Writer" --message "Nuclear energy" --stream
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure local src/ is importable without editable install
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from dotenv import load_dotenv, find_dotenv  # type: ignore
from langchain_core.messages import HumanMessage, BaseMessage  # type: ignore

# Import graphs
from open_deep_research.essay_writer import essay_writer  # type: ignore
from open_deep_research.deep_researcher import deep_researcher  # type: ignore
from open_deep_research.ghostwriter import ghostwriter  # type: ignore


GRAPHS = {
    "Essay Writer": essay_writer,
    "Ghostwriter": ghostwriter,
    "Deep Researcher": deep_researcher,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a local LangGraph graph programmatically")
    p.add_argument("--graph", default="Essay Writer", choices=list(GRAPHS.keys()))
    p.add_argument("--message", help="Prompt/message content")
    p.add_argument("--input-file", help="Read prompt from file")
    p.add_argument("--thread-id", default="cli-session", help="Thread/session id for state tracking")
    p.add_argument("--stream", action="store_true", help="Stream node events")
    p.add_argument("--raw-events", action="store_true", help="Print raw event dicts when streaming")
    p.add_argument("--output", help="Write final output to this file (final_report or final_document)")
    # Ghostwriter style options (used only when graph == "Ghostwriter")
    p.add_argument("--style-guide", help="Inline style guide text to apply (Ghostwriter)")
    p.add_argument("--style-file", help="Path to a file containing the style guide (Ghostwriter)")
    return p.parse_args()


def read_prompt(args: argparse.Namespace) -> str:
    if args.message:
        return args.message
    if args.input_file:
        return Path(args.input_file).read_text(encoding="utf-8")
    # stdin if piped
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Provide --message or --input-file or pipe stdin")


def ensure_out_dir(path: Optional[str]) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_output_if_any(final_state: Dict[str, Any], output_path: Optional[str]) -> None:
    # Support both essay/deep_researcher (final_report) and ghostwriter (final_document)
    final_text = (final_state or {}).get("final_report") or (final_state or {}).get("final_document")
    if output_path:
        ensure_out_dir(output_path)
        Path(output_path).write_text(final_text or "", encoding="utf-8")
        print(f"Saved output to {output_path}")
    else:
        if final_text:
            print(final_text)
        else:
            # If no final output (e.g., clarification), print last assistant message
            messages: List[BaseMessage] = (final_state or {}).get("messages", [])  # type: ignore
            if messages:
                print(messages[-1].content)


async def run_stream(graph, input_payload: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
    # Stream node-level events; print a compact log and return final state
    final_state: Dict[str, Any] | None = None
    async for event in graph.astream_events(input_payload, config={"configurable": {"thread_id": thread_id}}, version="v2"):
        etype = event.get("event")
        name = event.get("name")
        data = event.get("data", {}) or {}
        if etype == "on_node_start":
            print(f"▶️  {name}")
        elif etype == "on_node_end":
            print(f"✅  {name}")
        elif etype == "on_tool_start":
            tname = data.get("name")
            print(f"🔧  tool:{tname}")
        elif etype == "on_tool_end":
            tname = data.get("name")
            print(f"🔚  tool:{tname}")
        elif etype == "on_chain_end":
            final_state = data.get("output")
    return final_state or {}


async def main_async() -> None:
    load_dotenv(find_dotenv(), override=False)
    args = parse_args()
    prompt = read_prompt(args)

    # Optional style guide (used for Ghostwriter only)
    style_text: Optional[str] = None
    if getattr(args, "style_guide", None):
        style_text = args.style_guide
    elif getattr(args, "style_file", None):
        style_path = Path(args.style_file)
        if not style_path.exists():
            raise SystemExit(f"Style file not found: {style_path}")
        style_text = style_path.read_text(encoding="utf-8")

    graph = GRAPHS[args.graph]

    # Build input payload
    input_payload: Dict[str, Any] = {"messages": [HumanMessage(content=prompt)]}
    if args.graph == "Ghostwriter" and style_text:
        input_payload["style_guide"] = style_text

    if args.stream:
        final_state = await run_stream(graph, input_payload, args.thread_id)
        write_output_if_any(final_state, args.output)
        return

    # Non-stream one-shot
    final_state: Dict[str, Any] = await graph.ainvoke(
        input_payload,
        config={"configurable": {"thread_id": args.thread_id}},
    )
    write_output_if_any(final_state, args.output)


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
