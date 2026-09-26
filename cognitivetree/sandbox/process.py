"""Child-process execution with bounded output capture.

``subprocess.run`` buffers a child's entire output before returning, so a
payload that prints gigabytes costs the host gigabytes even when only the
first few kilobytes are kept. The runner here drains both pipes as the child
writes, keeps at most ``capture_bytes`` of each, and discards the rest while
continuing to read, so the child never blocks on a full pipe and host memory
stays bounded by the cap rather than by the payload.
"""

from __future__ import annotations

import contextlib
import subprocess
import threading
from dataclasses import dataclass
from typing import IO

_CHUNK_BYTES = 64 * 1024

# How long to wait for the pipe readers after the child has exited or been
# killed. A payload that spawned its own children can keep the pipes open
# past its parent's death; the readers are daemons, so giving up on them
# leaks nothing but a thread blocked on a pipe that closes eventually.
_READER_GRACE_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class BoundedRun:
    """Outcome of one bounded child-process run.

    ``returncode`` is ``None`` when the child was killed at the deadline.
    The ``*_truncated`` flags report that the stream emitted more than was
    kept.
    """

    returncode: int | None
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool

    @property
    def timed_out(self) -> bool:
        return self.returncode is None


class _BoundedReader(threading.Thread):
    """Drains one pipe, keeping only its first ``cap`` bytes."""

    def __init__(self, stream: IO[bytes], cap: int) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._cap = cap
        self._chunks: list[bytes] = []
        self._kept = 0
        self.truncated = False

    def run(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            while chunk := self._stream.read(_CHUNK_BYTES):
                room = self._cap - self._kept
                if room > 0:
                    kept = chunk[:room]
                    self._chunks.append(kept)
                    self._kept += len(kept)
                if len(chunk) > room:
                    self.truncated = True

    @property
    def data(self) -> bytes:
        return b"".join(self._chunks)


def run_bounded(
    command: list[str],
    stdin: bytes,
    timeout: float,
    capture_bytes: int,
) -> BoundedRun:
    """Runs ``command`` to completion or ``timeout``, capping captured output.

    Raises :class:`OSError` when the child cannot be spawned, as
    :func:`subprocess.run` would. A child still running at the deadline is
    killed, and whatever it emitted up to that point is returned.
    """
    if capture_bytes < 1:
        raise ValueError("capture_bytes must be a positive integer")
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin and process.stdout and process.stderr
    readers = [
        _BoundedReader(process.stdout, capture_bytes),
        _BoundedReader(process.stderr, capture_bytes),
    ]
    for reader in readers:
        reader.start()
    feeder = threading.Thread(target=_feed, args=(process.stdin, stdin), daemon=True)
    feeder.start()

    try:
        returncode: int | None = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        returncode = None
    finally:
        for reader in readers:
            reader.join(timeout=_READER_GRACE_SECONDS)

    out, err = readers
    return BoundedRun(
        returncode=returncode,
        stdout=out.data,
        stderr=err.data,
        stdout_truncated=out.truncated,
        stderr_truncated=err.truncated,
    )


def _feed(pipe: IO[bytes], data: bytes) -> None:
    """Writes the payload's stdin, tolerating a child that stops reading."""
    with contextlib.suppress(BrokenPipeError, OSError):
        if data:
            pipe.write(data)
    with contextlib.suppress(OSError):
        pipe.close()
