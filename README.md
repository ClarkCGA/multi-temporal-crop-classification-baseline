# gfm-segmentation-baseline
Baseline model for crop type segmentation as part of the GFM downstream task evaluations.

IBM NASA Foundation Model for Earth Observations (GFM) is [here](https://huggingface.co/ibm-nasa-geospatial)
The dataset used for training is [here](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification)


## Steps to run the code using docker:

**step 1-** Change directory to an empty folder in your machine and clone the branch.
```
$ cd /to_empty/dir/on_host/
$ git clone -b <branch_name> <branch_url>
```
Example:
```
git clone -b dev_datasetfusion https://github.com/ClarkCGA/gfm-segmentation-baseline.git
```

**step 2-** Make sure the docker software is running and build the docker image from the 'dockerfile' instructions.
```
docker build -t <image_name>:<tag> .
```

**step 3-** create a working container
```
docker run --gpus all -it -p 8888:8888 -v <path/to/the/cloned-repo/on-host>:/home/workdir -v <path/to/the/dataset/on-host>:/home/data  <image_name>
```
Example:
```
docker run --gpus all -it -p 8888:8888 -v "$(pwd)":/home/workdir -v /mnt/c/My_documents/summer_project/task1_baseline/dataset:/home/data  baseline_semseg_pytorch:v2
```

This command will create a container based on the specified docker image and starts a jupyterLab session. Type `localhost:8888` in your browser and copy the provided token from the terminal to open the jupyterlab.

To interact with the code:

**Option 1 -- step-by-step from a jupyter notebook (Recommended to get familiar with the pipeline)**
    
Open the jupyter notebook located at "notebooks/main.ipynb". 
Modify the "default_config.yaml" or create your own config file and run the cells as explained in the notebook.

**Option 2 -- Use the CLI** (To be completed)


