source /opt/ros/humble/setup.bash
# tracker 组件在 dlopen 时直接链接此 GXF 扩展；其目录默认不在动态链接器路径中。
compositor_lib=$(find /opt/ros/humble -name libgxf_isaac_message_compositor.so -print -quit 2>/dev/null)
if [[ -n "$compositor_lib" ]]; then
  compositor_dir=$(dirname -- "$compositor_lib")
  export LD_LIBRARY_PATH="$compositor_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
  printf 'WARNING: libgxf_isaac_message_compositor.so not found in /opt/ros/humble\n' >&2
fi
unset compositor_lib compositor_dir
PS1='\u@IssacFP:\w\$ '
