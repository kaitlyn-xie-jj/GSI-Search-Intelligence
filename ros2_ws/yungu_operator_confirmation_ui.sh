#!/usr/bin/env bash
set -eo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${1:?container name required}"
SEARCH_LOG="${2:?search log path required}"

clear
echo "Yungu 目标人工确认"
echo "90°俯视画面位于 Gazebo 的 Nadir Confirmation Camera 窗口。"
echo "等待无人机发现目标并悬停……"

handled=0
while true; do
    count="$(grep -c 'OPERATOR CONFIRMATION REQUIRED' "${SEARCH_LOG}" 2>/dev/null || true)"
    if (( count > handled )); then
        echo
        echo "============================================================"
        echo "检测到车辆候选，无人机已在目标上方悬停。"
        echo "请查看 Gazebo 中的 90° Nadir Confirmation Camera 窗口。"
        echo "============================================================"
        while true; do
            read -r -p "这是目标车辆吗？[y=是 / n=否]: " answer
            case "${answer,,}" in
                y|yes|correct) message=yes; break ;;
                n|no|wrong|incorrect) message=no; break ;;
                *) echo "请输入 y 或 n。" ;;
            esac
        done
        docker exec "${CONTAINER_NAME}" bash -lc \
            "source /opt/ros/humble/setup.bash; source /tmp/GSI/ros2_ws/install/setup.bash; ros2 topic pub --once /gsi/operator_confirmation std_msgs/msg/String \"{data: '${message}'}\""
        handled="${count}"
        echo "确认结果已发送：${message}"
        echo "等待下一次候选或搜索结束……"
    fi
    if docker exec "${CONTAINER_NAME}" bash -lc \
        "grep -q '\"event\": \"outcome\"' /tmp/GSI/results/yungu2030_sensor_validation/search_trace.jsonl 2>/dev/null"; then
        echo
        echo "搜索已结束。按 Enter 关闭窗口。"
        read -r || true
        exit 0
    fi
    sleep 1
done
