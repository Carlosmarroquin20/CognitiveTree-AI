"""Delivery of payload code over stdin instead of the command line.

Passing code as a ``-c`` argument ties its size to the platform's command
line limit, about 32k characters on Windows, where a longer payload fails to
spawn at all. The executors instead launch a small fixed bootstrap and stream
the code through stdin ahead of the payload's own input: a decimal byte
length, a newline, the UTF-8 code, then the stdin the request supplied.

The bootstrap reads exactly the framed code and leaves the rest of stdin in
the interpreter's buffer, so ``input()`` and ``sys.stdin`` see only the
payload's input. It compiles the code as ``<string>`` and runs it in
``__main__``, and its exception hook drops the bootstrap's own frames, so
tracebacks, exit codes, and the stderr the critics parse match what ``-c``
produced. Code reading the raw descriptor with ``os.read(0, ...)`` may miss
input the bootstrap's buffered read already consumed.
"""

from __future__ import annotations

BOOTSTRAP = """\
def _load():
    import sys
    payload = []
    default_hook = sys.excepthook

    def hook(kind, value, tb):
        while tb is not None and tb.tb_frame.f_code not in payload:
            tb = tb.tb_next
        # The default hook prints the exception's own traceback, not the
        # argument, so the trimmed chain must be attached to the exception.
        default_hook(kind, value.with_traceback(tb), tb)

    sys.excepthook = hook
    size = int(sys.stdin.buffer.readline())
    source = sys.stdin.buffer.read(size).decode("utf-8")
    del globals()["_load"]
    payload.append(compile(source, "<string>", "exec"))
    return payload[0]


exec(_load())
"""


def frame_payload(code: str, stdin: str) -> bytes:
    """Encodes ``code`` and the payload's ``stdin`` as the bootstrap expects."""
    body = code.encode("utf-8")
    return b"%d\n" % len(body) + body + stdin.encode("utf-8")
