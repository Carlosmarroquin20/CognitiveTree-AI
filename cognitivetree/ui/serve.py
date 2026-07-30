"""Command-line entry point for the streaming interface.

Examples:
    # Deterministic reference scenario (no model required)
    python -m cognitivetree.ui.serve

    # Open-source model behind an OpenAI-compatible endpoint
    python -m cognitivetree.ui.serve --backend llm \\
        --base-url http://localhost:11434/v1 --model llama3.3 \\
        --task "Implement a run-length encoder as encode(text)." \\
        --harness-file checks.py

    # Re-stream a saved run archive for offline inspection
    python -m cognitivetree.ui.serve --backend replay --archive runs/timed-out.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cognitivetree.config import SearchConfig
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
        choices=("reference", "llm-demo", "llm", "replay"),
        default="reference",
        help=(
            "reference: deterministic demo scenario; "
            "llm-demo: LLM adapter stack driven by a scripted client (no model); "
            "llm: live OpenAI-compatible endpoint; "
            "replay: re-stream a saved run archive"
        ),
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
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help=(
            "global wall-clock budget for the whole search; the run stops "
            "with outcome 'timed_out' once it elapses (default: unbounded)"
        ),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="run archive to re-stream; required by the 'replay' backend",
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=None,
        help=(
            "replay pacing factor; omit for instant replay, 1.0 to reproduce "
            "the run's original timing, 2.0 for twice that pace"
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    return parser


def session_factory_from_args(args: argparse.Namespace):
    """Builds the per-connection session factory selected by the CLI."""
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise SystemExit("--max-seconds must be positive")
    if args.replay_speed is not None and args.replay_speed <= 0:
        raise SystemExit("--replay-speed must be positive")

    if args.backend == "replay":
        from cognitivetree.persistence import ArchiveFormatError, ReplaySession, load_run

        if args.archive is None:
            raise SystemExit("backend 'replay' requires --archive")
        try:
            # Loading up front turns a bad path or corrupt document into an
            # immediate startup failure instead of a broken first request.
            archive = load_run(args.archive)
        except (OSError, ArchiveFormatError) as exc:
            raise SystemExit(f"cannot read archive {args.archive}: {exc}") from exc
        return lambda: ReplaySession(archive, speed=args.replay_speed)

    if args.backend == "reference":
        return lambda: build_reference_session(max_wall_seconds=args.max_seconds)
    if args.backend == "llm-demo":
        from cognitivetree.llm.demo import build_offline_session

        return lambda: build_offline_session(max_wall_seconds=args.max_seconds)
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
        config=SearchConfig(seed=None, max_wall_seconds=args.max_seconds),
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
