#!/bin/bash
docker run --gpus all -v `pwd`:/app/project/ -v ~/waymo:/waymo --network=host -ti project-dev jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.password=''
