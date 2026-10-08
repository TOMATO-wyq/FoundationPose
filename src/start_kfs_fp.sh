#!/usr/bin/env bash
# Interactive user launcher: build isolated image if needed, then run full KFS FP.
set -euo pipefail
kfs_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
kfs_mode=${1:-zed2i}
case "$kfs_mode" in
  -h|--help)
    echo 'Usage: bash src/start_kfs_fp.sh {zed2i|d435i|offline} [FP arguments]'
    echo 'Start the camera ROS driver and camera_ros_source.py first for live modes.'
    echo 'The first run builds foundationpose:kfs-sam2; sudo may ask for your password.'
    exit 0 ;;
  zed2i|d435i|offline) shift || true ;;
  *) echo 'Expected zed2i, d435i or offline' >&2; exit 2 ;;
esac
kfs_mask_mode=sam
kfs_scan=("$@")
for ((kfs_i=0; kfs_i<${#kfs_scan[@]}; kfs_i++)); do
  case "${kfs_scan[kfs_i]}" in
    --mask_mode|--mask-mode) kfs_mask_mode=${kfs_scan[kfs_i+1]:-} ;;
    --mask_mode=*|--mask-mode=*) kfs_mask_mode=${kfs_scan[kfs_i]#*=} ;;
    --mask-source) [[ ${kfs_scan[kfs_i+1]:-} != yolo ]] || kfs_mask_mode=yolo ;;
    --mask-source=yolo) kfs_mask_mode=yolo ;;
  esac
done
case "$kfs_mask_mode" in
  sam) python3 "$kfs_root/src/setup_kfs_sam2.py" --verify-only ;;
  yolo) ;;
  *) echo 'Expected --mask_mode sam or yolo' >&2; exit 2 ;;
esac
if [[ "$kfs_mode" != offline && ! -S "$kfs_root/outputs/live_rgbd.sock" ]]; then
  echo "Start the host RGB-D source in another terminal: python3 $kfs_root/src/camera_ros_source.py --camera $kfs_mode" >&2
  exit 1
fi
kfs_docker=(docker)
if ! docker info >/dev/null 2>&1; then kfs_docker=(sudo docker); fi
kfs_image=foundationpose:kfs-sam2
if [[ "$kfs_mask_mode" == yolo ]]; then kfs_image=foundationpose:kfs-yolo; fi
if ! "${kfs_docker[@]}" image inspect "$kfs_image" >/dev/null 2>&1; then
  echo "Building $kfs_image (first run)..."
  if [[ "$kfs_mask_mode" == yolo ]]; then
    "${kfs_docker[@]}" build --network=host -t "$kfs_image" \
      -f "$kfs_root/src/Dockerfile.kfs-yolo" "$kfs_root/src"
  else
  "${kfs_docker[@]}" build --network=host \
    --build-context "sam2=$kfs_root/third_party/sam2" \
    -t "$kfs_image" -f "$kfs_root/src/Dockerfile.kfs" "$kfs_root/src"
  fi
fi
kfs_args=(--rm -it --gpus all --shm-size=4g --network=host
  --user "$(id -u):$(id -g)"
  --mount type=bind,src=/etc/passwd,dst=/etc/passwd,readonly
  --mount type=bind,src=/etc/group,dst=/etc/group,readonly
  -e HOME=/tmp -e MPLCONFIGDIR=/tmp/matplotlib
  -e PYTHONPATH=/workspace/6Dpose/third_party/sam2:/workspace/6Dpose/src
  -v "$kfs_root:/workspace/6Dpose" -w /workspace/6Dpose/FoundationPose)
kfs_headless=()
if [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
  kfs_args+=(-e "DISPLAY=$DISPLAY" -e QT_X11_NO_MITSHM=1
    --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix)
  kfs_authority=${XAUTHORITY:-$HOME/.Xauthority}
  if [[ -f "$kfs_authority" ]]; then
    kfs_args+=(-e XAUTHORITY=/tmp/.kfs.Xauthority
      --mount "type=bind,src=$kfs_authority,dst=/tmp/.kfs.Xauthority,readonly")
  fi
else
  kfs_headless=(--no-display)
fi
kfs_output="/workspace/6Dpose/outputs/kfs_${kfs_mode}_$(date +%Y%m%d_%H%M%S_%N)"
kfs_command=(python -u ../src/run_foundationpose_live.py --camera "$kfs_mode" --save-poses)
if [[ "$kfs_mode" == offline ]]; then
  kfs_command=(python -u ../src/run_foundationpose_zed.py --no-video)
fi
kfs_command+=(--mask_mode "$kfs_mask_mode" --depth-refinement
  --segmentation-config ../src/config/kfs_blue.json --output "$kfs_output")
if [[ "$kfs_mask_mode" == sam ]]; then
  kfs_command+=(--sam-checkpoint ../third_party/sam2/checkpoints/sam2.1_hiera_small.pt)
else
  kfs_command+=(--yolo-weights ../weights/kfs_yolo_seg.pt)
fi
echo "Output: $kfs_output"
exec "${kfs_docker[@]}" run "${kfs_args[@]}" "$kfs_image" \
  "${kfs_command[@]}" "${kfs_headless[@]}" "$@"
