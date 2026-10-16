# Isaac ROS 开发构建脚本

适用于 Jetson 或 x86_64 平台：

`run_dev.sh` 会创建一个已安装 ROS 2 的开发环境。默认情况下，容器内的 `/workspaces/isaac_ros-dev` 会映射到主机上的 `~/workspaces/isaac_ros-dev`；如果主机上不存在该目录，则映射到调用脚本时所在的当前目录。也可以运行脚本并指定目录，显式设置容器要映射的主机目录：

```bash
run_dev.sh -d <工作区路径>
```
