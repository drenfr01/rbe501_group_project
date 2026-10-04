FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# Install Miniforge directly (avoids ghcr.io devcontainer Features auth/registry issues).
RUN set -eux; \
    arch="$(uname -m)"; \
    case "${arch}" in \
      x86_64) installer=Miniforge3-Linux-x86_64.sh ;; \
      aarch64|arm64) installer=Miniforge3-Linux-aarch64.sh ;; \
      *) echo "Unsupported architecture: ${arch}" >&2; exit 1 ;; \
    esac; \
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/${installer}" -O /tmp/miniforge.sh; \
    bash /tmp/miniforge.sh -b -p /home/vscode/miniforge3; \
    rm /tmp/miniforge.sh; \
    chown -R vscode:vscode /home/vscode/miniforge3

USER vscode
