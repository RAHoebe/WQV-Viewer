# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    QT_X11_NO_MITSHM=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    VNC_PASSWORD=viewonly \
    RESOLUTION=1280x800 \
    NOVNC_PORT=6080 \
    VNC_PORT=5901 \
    AUTO_LAUNCH_VIEWER=1

ENV JUPYTER_ENABLE=1 \
    JUPYTER_PORT=8888 \
    JUPYTER_TOKEN="" \
    JUPYTER_ROOT=/data

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-distutils python3-pip python3-dev \
        build-essential pkg-config \
        xfce4 xfce4-terminal \
        tigervnc-standalone-server tigervnc-common tigervnc-tools \
        novnc websockify \
        python3-numpy \
        curl ca-certificates git xdg-utils \
        libglib2.0-0 libsm6 libice6 libx11-6 libx11-xcb1 libxext6 libxrender1 libxi6 libxrandr2 libxtst6 \
        libxcomposite1 libxcursor1 libxdamage1 libxfixes3 libfontconfig1 libfreetype6 \
        libxkbcommon0 libxkbcommon-x11-0 \
        libgl1 libegl1 libegl-mesa0 libopengl0 \
        libxcb1 libxcb-render0 libxcb-render-util0 libxcb-shape0 libxcb-shm0 libxcb-xfixes0 libxcb-randr0 \
        libxcb-image0 libxcb-keysyms1 libxcb-icccm4 libxcb-sync1 libxcb-xinerama0 libxcb-xkb1 libxcb-util1 \
        libxcb-cursor0 xauth && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    python3 -m venv /opt/wqv && \
    /opt/wqv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

ENV PATH=/opt/wqv/bin:$PATH

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir .[dev] jupyterlab numpy && \
    python -m compileall wqv_viewer wqv_upscale_trainer

RUN mkdir -p /root/.vnc /data /usr/share/applications /root/Desktop

RUN cat <<'EOF' >/usr/share/applications/wqv-viewer.desktop
[Desktop Entry]
Type=Application
Name=WQV Viewer
Comment=Launch the WQV Wristcam Viewer GUI
Exec=python -m wqv_viewer
Icon=applications-graphics
Terminal=false
Categories=Graphics;
EOF

RUN cat <<'EOF' >/usr/share/applications/wqv-upscale-trainer.desktop
[Desktop Entry]
Type=Application
Name=WQV Upscale Trainer
Comment=Open a terminal running the trainer CLI
Exec=xfce4-terminal --title="WQV Upscale Trainer" -e "bash -lc 'wqv-upscale-trainer; exec bash'"
Icon=utilities-terminal
Terminal=false
Categories=Utility;
EOF

RUN install -m 755 /usr/share/applications/wqv-viewer.desktop /root/Desktop/wqv-viewer.desktop && \
    install -m 755 /usr/share/applications/wqv-upscale-trainer.desktop /root/Desktop/wqv-upscale-trainer.desktop

RUN cat <<'EOF' >/usr/share/novnc/index.html
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta http-equiv="refresh" content="0; url=vnc.html?autoconnect=1&resize=scale" />
        <title>WQV Viewer Desktop</title>
        <style>
            body { font-family: sans-serif; background: #111; color: #eee; display: flex; align-items: center; justify-content: center; height: 100vh; }
            a { color: #7ec0ff; }
        </style>
    </head>
    <body>
        <p>Launching the virtual desktop… <a href="vnc.html?autoconnect=1&resize=scale">click here</a> if you are not redirected.</p>
    </body>
</html>
EOF

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 6080 8888
ENTRYPOINT ["/entrypoint.sh"]
CMD []
