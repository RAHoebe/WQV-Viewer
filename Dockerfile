# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/pytorch:23.07-py3

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    QT_X11_NO_MITSHM=1 \
    QTWEBENGINE_DISABLE_SANDBOX=1 \
    SHELL=/bin/bash \
    DISPLAY=:1 \
    CONDA_ENV_NAME=wqv \
    VNC_PORT=5901 \
    NOVNC_PORT=6080 \
    VNC_PASSWORD=wqv \
    VNC_DISABLE_PASSWORD=0 \
    RESOLUTION=1920x1080 \
    VNC_DEPTH=24 \
    AUTO_LAUNCH_VIEWER=0 \
    JUPYTER_ENABLE=1 \
    JUPYTER_PORT=8888 \
    JUPYTER_TOKEN="" \
    JUPYTER_ROOT=/home/wqv \
    PATH=/opt/conda/bin:$PATH

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        dbus-x11 \
        xfce4 \
        xfce4-goodies \
        xfce4-terminal \
        xterm \
        x11-xserver-utils \
        tigervnc-standalone-server \
        tigervnc-common \
        novnc \
        websockify \
        sudo \
        curl \
        wget \
        git \
        xz-utils \
        fonts-dejavu \
        fonts-liberation \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libxi6 \
        libxrandr2 \
        libxtst6 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxfixes3 \
        libxkbcommon0 \
        libxkbcommon-x11-0 \
        libfontconfig1 \
        libnss3 \
        libglu1-mesa \
        libpulse0 \
        ca-certificates && \
    apt-get autoremove --purge -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

ARG MINIFORGE_VERSION=25.9.1-0
RUN curl -fsSL "https://github.com/conda-forge/miniforge/releases/download/Mambaforge-${MINIFORGE_VERSION}/Mambaforge-${MINIFORGE_VERSION}-Linux-x86_64.sh" -o /tmp/mambaforge.sh && \
    bash /tmp/mambaforge.sh -b -p /opt/conda && \
    rm -f /tmp/mambaforge.sh && \
    /opt/conda/bin/conda clean -afy

RUN useradd -m -s /bin/bash wqv && \
    echo "wqv ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/wqv && \
    chmod 0440 /etc/sudoers.d/wqv && \
    mkdir -p /home/wqv/Desktop /opt/icons /tmp/runtime-wqv && \
    chown -R wqv:wqv /home/wqv /opt/icons /tmp/runtime-wqv && \
    chmod 700 /tmp/runtime-wqv

COPY --chown=wqv:wqv . /opt/wqv-viewer

WORKDIR /opt/wqv-viewer
SHELL ["/bin/bash", "-c"]
RUN /opt/conda/bin/conda update -n base conda -y && \
    /opt/conda/bin/conda install -n base -y pip && \
    /opt/conda/bin/conda create -y -n "${CONDA_ENV_NAME}" python=3.11 && \
    /opt/conda/bin/conda run -n "${CONDA_ENV_NAME}" python -m pip install --upgrade pip setuptools wheel && \
    /opt/conda/bin/conda run -n "${CONDA_ENV_NAME}" python -m pip install --no-cache-dir .[dev] && \
    /opt/conda/bin/conda run -n "${CONDA_ENV_NAME}" python -m compileall wqv_viewer wqv_upscale_trainer && \
    /opt/conda/bin/conda clean -afy

RUN mkdir -p /home/wqv/.config/QtProject && \
    printf '[Paths]\nPrefix=/opt/conda/envs/%s\n' "$CONDA_ENV_NAME" > /home/wqv/.config/QtProject/qtpaths

RUN cat <<'EOF' >/opt/icons/start_wqv_viewer.sh
#!/usr/bin/env bash
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-wqv}"
python -m wqv_viewer "$@"
EOF
RUN chmod +x /opt/icons/start_wqv_viewer.sh

RUN cat <<'EOF' >/opt/icons/start_wqv_trainer.sh
#!/usr/bin/env bash
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-wqv}"
wqv-upscale-trainer "$@"
exec bash
EOF
RUN chmod +x /opt/icons/start_wqv_trainer.sh

RUN cat <<'EOF' >/home/wqv/Desktop/wqv-viewer.desktop
[Desktop Entry]
Type=Application
Version=1.0
Name=WQV Viewer
Comment=Launch the WQV Wristcam viewer GUI
Exec=/opt/icons/start_wqv_viewer.sh
Icon=applications-graphics
Terminal=false
Categories=Graphics;
EOF && \
    cat <<'EOF' >/home/wqv/Desktop/wqv-trainer.desktop
[Desktop Entry]
Type=Application
Version=1.0
Name=WQV Upscale Trainer
Comment=Open a terminal for the WQV NeoSR trainer
Exec=xfce4-terminal --title="WQV Upscale Trainer" --command="/opt/icons/start_wqv_trainer.sh"
Icon=utilities-terminal
Terminal=false
Categories=Utility;
EOF && \
    chmod +x /home/wqv/Desktop/wqv-viewer.desktop /home/wqv/Desktop/wqv-trainer.desktop && \
    chown -R wqv:wqv /home/wqv/Desktop

RUN cat <<'EOF' >/usr/share/novnc/index.html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="refresh" content="0; url=vnc.html?autoconnect=1&resize=scale" />
    <title>WQV Viewer Desktop</title>
    <style>
      body { font-family: sans-serif; background: #111; color: #eee; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
      a { color: #7ec0ff; text-decoration: none; }
    </style>
  </head>
  <body>
    <p>Launching the desktop… <a href="vnc.html?autoconnect=1&resize=scale">click here</a> if you are not redirected.</p>
  </body>
</html>
EOF

COPY docker/entrypoint.sh /usr/local/bin/start-wqv.sh
RUN chmod +x /usr/local/bin/start-wqv.sh && chown wqv:wqv /usr/local/bin/start-wqv.sh

RUN chown -R wqv:wqv /opt/wqv-viewer

WORKDIR /home/wqv
USER wqv
ENV HOME=/home/wqv \
    XDG_RUNTIME_DIR=/tmp/runtime-wqv

EXPOSE 5901 6080 8888
ENTRYPOINT ["/usr/local/bin/start-wqv.sh"]
CMD []
