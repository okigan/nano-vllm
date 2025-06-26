#!/bin/bash
set -e

# Detect JetPack version from /etc/nv_tegra_release
JP_VERSION_RAW=$(grep -o 'R[0-9]*' /etc/nv_tegra_release | head -n1 | tr -d 'R')
JP_REVISION=$(grep -o 'REVISION: [0-9]*\.[0-9]*' /etc/nv_tegra_release | head -n1 | awk '{print $2}')

# Compose JetPack version string for NVIDIA download URLs
if [[ -n "$JP_VERSION_RAW" && -n "$JP_REVISION" ]]; then
    JP_VERSION="${JP_VERSION_RAW}${JP_REVISION//./}"
else
    echo "Could not detect JetPack version. Please set JP_VERSION manually."
    exit 1
fi

echo "Detected JetPack version: $JP_VERSION"

# Set PyTorch version and wheel URL based on JetPack version (update as needed)
# Example for JetPack 6.0.3 (JP_VERSION=3643):
if [[ "$JP_VERSION" == "3643" ]]; then
    # TORCH_WHL_URL="https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl"
    TORCH_WHL_URL="https://nvidia.box.com/shared/static/zvultzsmd4iuheykxy17s4l2n91ylpl8.whl"
    TORCH_WHL="torch-2.3.0-cp310-cp310-linux_aarch64.whl"
else
    echo "JetPack version $JP_VERSION is not explicitly supported in this script. Please update the script with the correct wheel URL for your JetPack version."
    exit 1
fi


# Uninstall any existing torch
pip uninstall -y torch torchvision torchaudio || true



# Download directory for wheels
DOWNLOAD_DIR="./build/download"
mkdir -p "$DOWNLOAD_DIR"
TORCH_WHL_PATH="$DOWNLOAD_DIR/$TORCH_WHL"

# Download and install PyTorch wheel if not already present
if [ ! -f "$TORCH_WHL_PATH" ]; then
    wget -O "$TORCH_WHL_PATH" "$TORCH_WHL_URL"
else
    echo "$TORCH_WHL_PATH already exists, skipping download."
fi
uv pip install "$TORCH_WHL_PATH"

# Verify CUDA availability
uv run python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"

echo "NVIDIA PyTorch installation complete."