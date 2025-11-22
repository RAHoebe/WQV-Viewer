# WQV Viewer Docker Image

This guide explains how to build and run a containerized environment for **WQV Viewer** and **wqv_upscale_trainer** with GPU support and a browser-based desktop session.

## Prerequisites
- Docker Engine 24+
- NVIDIA driver and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- (Optional) Docker Hub account for publishing images

## Build the image
```bash
docker build -t wqv-viewer:latest -f Dockerfile .
```

## Run locally
```bash
docker run --gpus all -p 6080:6080 -p 5901:5901 -p 8888:8888 -e VNC_PASSWORD=yourpass -v C:\path\to\data:/data wqv-viewer
```

### Launch the viewer (VNC / port 6080)
1. Open a browser to **http://localhost:6080** and authenticate with the `VNC_PASSWORD` you supplied.
2. The container auto-launches the WQV Viewer window inside the session. Use it directly; if you close it, the XFCE desktop appears so you can relaunch the app from **Applications ▸ Graphics ▸ WQV Viewer** or by running `python -m wqv_viewer` in an XFCE terminal.

### Run the trainer (JupyterLab / port 8888)
1. Visit **http://localhost:8888** in your browser.
2. From the JupyterLab launcher choose **Terminal**.
3. In the terminal run:
   ```bash
   wqv-upscale-trainer
   ```
   The CLI will execute inside that shell; keep the window open to monitor training output.

The bind mount in the `docker run` command exposes your host dataset directory at `/data` inside the container. Adjust the Windows path so it matches your setup (for other drives use, for example, `D:\datasets`). Both the desktop apps and JupyterLab can read/write there directly.

## Push to Docker Hub
```bash
docker tag wqv-viewer:latest <username>/wqv-viewer:latest
docker login
docker push <username>/wqv-viewer:latest
```

## Environment variables
- `VNC_PASSWORD`: password for the VNC session (default `wqv`)
- `VNC_DISABLE_PASSWORD`: set to `1` to allow passwordless access (uses insecure `-SecurityTypes None`)
- `RESOLUTION`: desktop resolution (default `1920x1080`)
- `AUTO_LAUNCH_VIEWER`: set to `0` to stop auto-starting the GUI
- `NOVNC_PORT`: change the exposed noVNC port (default `6080`)
- `JUPYTER_ENABLE`: set to `0` to skip launching JupyterLab (default `1`)
- `JUPYTER_PORT`: change the exposed JupyterLab port (default `8888`)
- `JUPYTER_TOKEN`: supply a token for Jupyter authentication (blank by default)
- `JUPYTER_ROOT`: working directory served by JupyterLab (default `/home/wqv`)

Mount Palm database directories or model folders into `/data` (or another mount point) using `-v` or compose volumes. Access notebooks under `/data` via JupyterLab, or use the desktop shortcuts inside the virtual environment.
