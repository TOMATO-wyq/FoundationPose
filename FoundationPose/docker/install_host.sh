#!/usr/bin/env bash
# Ubuntu host setup, based on Docker and NVIDIA's official apt instructions.
# Run in your own terminal: bash docker/install_host.sh
set -euo pipefail
source /etc/os-release
if [[ "$ID" != ubuntu || "${VERSION_ID}" != 22.04 ]]; then
  echo 'This script was prepared for Ubuntu 22.04; inspect it before using another OS.' >&2
  exit 1
fi
if [[ "$EUID" == 0 ]]; then
  echo 'Run as your normal user; the script invokes sudo where needed.' >&2
  exit 1
fi
sudo -v
nvidia-smi

# Do not automatically remove packages used by other container installations.
for pkg in docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc; do
  if [[ "$(dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null || true)" == 'install ok installed' ]]; then
    echo "Conflicting package found: $pkg. Review existing containers before proceeding." >&2
    exit 1
  fi
done

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
task_tmp=$(mktemp -d)
trap 'rm -rf "$task_tmp"' EXIT
curl --retry 3 -fsSL https://download.docker.com/linux/ubuntu/gpg -o "$task_tmp/docker.asc"
sudo install -m 0644 "$task_tmp/docker.asc" /etc/apt/keyrings/docker.asc
cat > "$task_tmp/docker.sources" <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: jammy
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo install -m 0644 "$task_tmp/docker.sources" /etc/apt/sources.list.d/docker.sources

curl --retry 3 -fsSL https://nvidia.github.io/libnvidia-container/gpgkey -o "$task_tmp/nvidia.asc"
gpg --batch --dearmor -o "$task_tmp/nvidia.gpg" "$task_tmp/nvidia.asc"
sudo install -m 0644 "$task_tmp/nvidia.gpg" /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl --retry 3 -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  -o "$task_tmp/nvidia.list"
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  "$task_tmp/nvidia.list" > "$task_tmp/nvidia-container-toolkit.list"
sudo install -m 0644 "$task_tmp/nvidia-container-toolkit.list" /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-container-toolkit
if sudo test -f /etc/docker/daemon.json; then
  sudo cp -a /etc/docker/daemon.json "/etc/docker/daemon.json.foundationpose-backup-$(date +%Y%m%d-%H%M%S)"
fi
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl enable --now docker
sudo systemctl restart docker
sudo docker run --rm hello-world
sudo docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu22.04 nvidia-smi
echo 'Docker and NVIDIA Container Toolkit installed; GPU passthrough verified.'
echo 'Docker commands currently use sudo. No user group membership was changed.'
