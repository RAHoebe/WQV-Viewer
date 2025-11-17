#!/usr/bin/env bash
set -euo pipefail

# Defaults may be overridden via environment variables.
VNC_DISPLAY=":1"
VNC_PORT="${VNC_PORT:-5901}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
VNC_PASSWORD="${VNC_PASSWORD:-viewonly}"
DISABLE_PASSWORD="${VNC_DISABLE_PASSWORD:-0}"
RESOLUTION="${RESOLUTION:-1280x800}"
DEPTH="${VNC_DEPTH:-24}"
AUTO_LAUNCH="${AUTO_LAUNCH_VIEWER:-1}"
JUPYTER_ENABLE="${JUPYTER_ENABLE:-1}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-}"
JUPYTER_ROOT="${JUPYTER_ROOT:-/data}"

export DISPLAY="${VNC_DISPLAY}"

mkdir -p /root/.vnc "${JUPYTER_ROOT}"

# Configure VNC password when enabled
SECURITY_OPTS=""
if [ "${DISABLE_PASSWORD}" = "1" ]; then
    echo "Starting VNC server without authentication"
    rm -f /root/.vnc/passwd
    SECURITY_OPTS="-SecurityTypes None"
else
    if [ ! -f /root/.vnc/passwd ]; then
        echo "Setting VNC password"
        printf "%s\n%s\n\n" "${VNC_PASSWORD}" "${VNC_PASSWORD}" | vncpasswd -f > /root/.vnc/passwd
        chmod 600 /root/.vnc/passwd
    fi
fi

# Ensure the X startup script exists
cat <<'EOF' > /root/.vnc/xstartup
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
exec startxfce4
EOF
chmod +x /root/.vnc/xstartup

cleanup() {
    if pgrep -x websockify >/dev/null 2>&1; then
        pkill -TERM websockify || true
    fi
    if pgrep -f "jupyter-lab" >/dev/null 2>&1; then
        pkill -TERM -f "jupyter-lab" || true
    fi
    if [ -f /root/.vnc/"$(hostname)${VNC_DISPLAY}".pid ]; then
        vncserver -kill "${VNC_DISPLAY}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

# Launch the VNC server
vncserver "${VNC_DISPLAY}" -geometry "${RESOLUTION}" -depth "${DEPTH}" ${SECURITY_OPTS}

# Bridge VNC to noVNC so we can connect via a browser
NOVNC_WEB=/usr/share/novnc

if command -v websockify >/dev/null 2>&1; then
    websockify --web "${NOVNC_WEB}" "${NOVNC_PORT}" localhost:"${VNC_PORT}" &
else
    "${NOVNC_WEB}/utils/novnc_proxy" --listen "${NOVNC_PORT}" --vnc localhost:"${VNC_PORT}" &
fi
WEBSOCKIFY_PID=$!

echo "noVNC available on http://localhost:${NOVNC_PORT}"

declare -a CHILD_PIDS=("${WEBSOCKIFY_PID}")

if [ "${JUPYTER_ENABLE}" = "1" ]; then
    echo "Starting JupyterLab on port ${JUPYTER_PORT}"
    jupyter lab \
        --ip=0.0.0.0 \
        --port="${JUPYTER_PORT}" \
        --no-browser \
        --ServerApp.token="${JUPYTER_TOKEN}" \
        --ServerApp.password='' \
        --ServerApp.allow_origin='*' \
        --ServerApp.disable_check_xsrf=True \
        --ServerApp.root_dir="${JUPYTER_ROOT}" \
        --ServerApp.open_browser=False \
        >/var/log/jupyter.log 2>&1 &
    CHILD_PIDS+=("$!")
    echo "JupyterLab available on http://localhost:${JUPYTER_PORT}"
fi

if [ "${AUTO_LAUNCH}" = "1" ]; then
    echo "Launching WQV Viewer"
    python -m wqv_viewer "$@" &
    CHILD_PIDS+=("$!")
fi

# Forward any additional commands when auto-launch disabled
if [ "${AUTO_LAUNCH}" != "1" ] && [ "$#" -gt 0 ]; then
    "$@" &
    CHILD_PIDS+=("$!")
fi

# Wait on child processes
while true; do
    if ! wait -n 2>/dev/null; then
        break
    fi
done
