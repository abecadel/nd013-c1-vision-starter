#!/bin/bash
docker build -t project-dev -f build/Dockerfile build/ && \
	
	
#docker run --gpus all -v `pwd`:/app/project/ -v ~/waymo:/waymo --network=host -ti project-dev jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.password=''

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v `pwd`:/app/project/ -v ~/waymo:/waymo -it --rm project-dev
