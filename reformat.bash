#!/bin/bash

fd -t f -e cpp -e cu -e hpp -e cuh -E StiffGIPC/muda -E MshHelper/MshIO  . MeshProcess/ MshHelper/ StiffGIPC/ | xargs clang-format -i
