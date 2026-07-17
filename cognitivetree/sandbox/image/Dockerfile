# Sandbox image for CognitiveTree-AI payload execution.
# The runtime surface is deliberately minimal: package managers are removed,
# no build toolchain is present, and the container runs as an unprivileged
# user. Runtime hardening (network, filesystem, capabilities, resource caps)
# is enforced by the executor's `docker run` profile, not by this image.

FROM python:3.12-slim

# Removes installer tooling so a compromised payload cannot pull dependencies.
RUN python -m pip uninstall --yes pip setuptools wheel \
    && rm -rf /root/.cache

# Dedicated unprivileged identity; the executor additionally forces
# --user 65534:65534 at run time for defense in depth.
RUN useradd --system --uid 10001 --no-create-home sandbox
USER sandbox

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /tmp

# No ENTRYPOINT: the executor supplies the full exec-form command.
CMD ["python", "-I", "-c", "print('cognitivetree sandbox ready')"]
