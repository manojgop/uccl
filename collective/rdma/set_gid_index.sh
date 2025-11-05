#!/bin/bash
if [ "$(hostname)" = "emrh01" ]; then
  export NCCL_IB_GID_INDEX=2
elif [ "$(hostname)" = "emrh02" ]; then
  export NCCL_IB_GID_INDEX=2
elif [ "$(hostname)" = "emrh03" ]; then
  export NCCL_IB_GID_INDEX=1
elif [ "$(hostname)" = "emrh04" ]; then
  export NCCL_IB_GID_INDEX=1
fi
exec "$@"

