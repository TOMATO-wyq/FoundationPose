#!/usr/bin/env bash
# Source after ROS setup. Keep inherited CycloneDDS network settings and add
# local unicast discovery, including when multicast is disabled and the only
# configured peers belong to another machine/network.
if [[ "${RMW_IMPLEMENTATION:-}" == rmw_cyclonedds_cpp ]]; then
  export CYCLONEDDS_URI="${CYCLONEDDS_URI:+${CYCLONEDDS_URI},}<CycloneDDS><Domain id=\"any\"><Discovery><Peers><Peer address=\"127.0.0.1\"/></Peers></Discovery></Domain></CycloneDDS>"
fi
