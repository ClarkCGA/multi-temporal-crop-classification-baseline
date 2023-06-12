# gfm-segmentation-baseline
Baseline model for crop type segmentation as part of the GFM downstream task evaluations

## Steps to run the code using docker:

step 1- Change directory to an empty folder in your machine and clone the branch.
```
$ cd /to_empty/dir/on_host/
$ git clone -b <branch_name> <branch_url>
```

step 2- Make sure the docker software is running and build the docker image from the 'dockerfile' instructions.
```
docker build -t <image_name>:<tag> .
```

step 3- create a working container
```
docker run --gpus all -it -p 8888:8888 -v <path/to/the/cloned-repo/on-host>:/home/workdir -v <path/to/the/dataset/on-host>:/home/data  <image_name>
```