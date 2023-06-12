# gfm-segmentation-baseline
Baseline model for crop type segmentation as part of the GFM downstream task evaluations

## Steps to run the code using docker:

**step 1-** Change directory to an empty folder in your machine and clone the branch.
```
$ cd /to_empty/dir/on_host/
$ git clone -b <branch_name> <branch_url>
```

**step 2-** Make sure the docker software is running and build the docker image from the 'dockerfile' instructions.
```
docker build -t <image_name>:<tag> .
```

**step 3-** create a working container
```
docker run --gpus all -it -p 8888:8888 -v <path/to/the/cloned-repo/on-host>:/home/workdir -v <path/to/the/dataset/on-host>:/home/data  <image_name>
```
This command will create a container based on the specified docker image and starts a jupyterLab session. Type `localhost:8888` in your browser and copy the provided token from the terminal to open the jupyterlab.

Two opens are provided to interact with the code:

**Option 1 -- step-by-step from a jupyter notebook (Recommended to get familiar with the pipeline)**
    Use the "notebooks/main.ipynb" jupyter notebook. Modify the "config" dictionary and run the cells as explained in the notebook.

**Option 2 -- Use the CLI**
    First update the parameters in the "config.yaml" file. Parameters are explained inside the file.
    
    Run the command below: