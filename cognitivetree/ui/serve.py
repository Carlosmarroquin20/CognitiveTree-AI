"""Command-line entry point for the streaming interface.

Examples:
    # Deterministic reference scenario (no model required)
    python -m cognitivetree.ui.serve

    # Open-source model behind an OpenAI-compatible endpoint
    python -m cognitivetree.ui.serve --backend llm \\
        --base-url http://localhost:11434/v1 --model llama3.3 \\
        --task "Implement a run-length encoder as encode(text)." \\
        --harness-file checks.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cognitivetree.session import (
    LlmSessionSpec,
    ReasoningSession,
    build_llm_session,
    build_reference_session,
)
from cognitivetree.ui.server import StreamingUiServer


def build_parser() -> argparse.ArgumentParser:
    """Declares the CLI surface."""
    parser = argparse.ArgumentParser(
        prog="cognitivetree-ui",
        description="Serves the CognitiveTree-AI live reasoning stream.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8732, help="bind port")
    parser.add_argument(
        "--backend",
        choices=("reference", "llm"),
        default="reference",
        help="reference: deterministic demo scenario; llm: OpenAI-compatible endpoint",
    )
    parser.add_argument("--base-url", help="endpoint root, e.g. http://localhost:11434/v1")
    parser.add_argument("--model", help="served model identifier, e.g. llama3.3")
    parser.add_argument("--task", help="task statement for the llm backend")
    parser.add_argument("--api-key", help="bearer token when the endpoint requires one")
    parser.add_argument(
        "--harness-file",
        type=Path,
        help="file with validation assertions appended to every payload",
    )
    parser.add_argument(
        "--llm-critic",
        action="store_true",
        help="chain an LLM critic behind the execution-trace critic",
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    return parser


def session_factory_from_args(args: argparse.Namespace):
    """Builds the per-connection session factory selected by the CLI."""
    if args.backend == "reference":
        return build_reference_session
    missing = [
        name
        for name, value in (
            ("--base-url", args.base_url),
            ("--model", args.model),
            ("--task", args.task),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            f"backend 'llm' requires {', '.join(missing)}"
        )
    harness = args.harness_file.read_text(encoding="utf-8") if args.harness_file else ""
    spec = LlmSessionSpec(
        task=args.task,
        base_url=args.base_url,
        model=args.model,
        validation_harness=harness,
        api_key=args.api_key,
        use_llm_critic=args.llm_critic,
    )

    def factory() -> ReasoningSession:
        return build_llm_session(spec)

    return factory


def main(argv: list[str] | None = None) -> None:
    """Parses arguments and serves until interrupted."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    server = StreamingUiServer((args.host, args.port), session_factory_from_args(args))
    print(f"CognitiveTree-AI streaming interface: {server.url}")
    print(f"backend: {args.backend} | stream endpoint: {server.url}stream")
    print("press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("server stopped")


if __name__ == "__main__":
    main()
