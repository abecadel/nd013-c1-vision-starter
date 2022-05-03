#!/bin/bash
docker build -t project-dev -f build/Dockerfile build/ && \
	
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v `pwd`:/app/project/ -v ~/waymo:/waymo --network=host -it --rm project-dev
