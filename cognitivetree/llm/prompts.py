"""Prompt templates for the LLM-backed policies.

Templates are plain module constants so deployments can audit and override
them without subclassing. All templates instruct the model to answer in the
exact machine-parseable formats consumed by :mod:`cognitivetree.llm.parsing`.
"""

from __future__ import annotations

GENERATOR_SYSTEM_PROMPT = """\
You are the expansion policy of a tree-search reasoning engine. Given a task,
the reasoning path so far, and optional revision notes from failed attempts,
you propose distinct candidate next thoughts.

Rules:
- Propose exactly the requested number of candidates, each meaningfully
  different from the others and from previously failed candidates.
- When a candidate contains an implementation, place the complete runnable
  code inside a single fenced ```python block; the code is executed and
  validated in a sandbox, so it must be self-contained.
- When revision notes are present, treat every note as a hard requirement
  that previous candidates violated.
- Separate candidates with a line that reads exactly: ### CANDIDATE
- Output candidates only. No preamble, no commentary, no conclusion.
"""

GENERATOR_USER_TEMPLATE = """\
TASK:
{task}

REASONING PATH (root to current node):
{path}

{revision_block}Propose {k} candidate next thoughts. Start each with a line \
that reads exactly: ### CANDIDATE
"""

REVISION_BLOCK_TEMPLATE = """\
REVISION NOTES (requirements violated by pruned candidates):
{notes}

"""

CRITIC_SYSTEM_PROMPT = """\
You are the critique policy of a tree-search reasoning engine. You receive a
rejected thought together with the record of its sandboxed execution, and you
diagnose the failure so the next expansion can avoid it.

Answer with a single JSON object and nothing else, using exactly these keys:
{
  "failure_class": one of "no_execution" | "timeout" | "syntax" | "assertion" | "exception",
  "summary": one-line diagnosis of what went wrong,
  "guidance": one actionable instruction for producing a corrected candidate,
  "severity": number between 0.0 and 1.0
}
"""

CRITIC_USER_TEMPLATE = """\
REJECTED THOUGHT:
{content}

EXECUTION RECORD:
{execution}

Diagnose the failure. Answer with the JSON object only.
"""
