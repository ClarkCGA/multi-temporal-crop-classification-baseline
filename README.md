# gfm-segmentation-baseline
Baseline model for crop type segmentation as part of the GFM downstream task evaluations.

IBM NASA Foundation Model for Earth Observations (GFM) is [here](https://huggingface.co/ibm-nasa-geospatial)
The dataset used for training is [here](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification). 


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

To interact with the pipeline:

** Open the jupyter notebook located at "notebooks/main.ipynb".
Modify the "default_config.yaml" or create your own config file and run the cells as explained in the notebook.**

The model weights trained on the dataset for 100 epochs with the parameters specified in the "default_config.yaml", is located in the "model_weights". How to use load and use the pre-trained model for zero-shot inference or warm-up training is explained in the notebook.

# Evaluation metrics of the pre-trained model:
![Confusion Matrix](./images/confusion_matrix.png)   
 
## Overall Metrics:

|Metric          |Value   |
|----------------|--------|
|Overall Accuracy|0.63056 |
|Mean Accuracy   |0.61915 |
|Mean IoU        |0.42086 |
|mean Precision  |0.57392 |
|mean Recall     |0.57492 |
|Mean F1 Score   |0.57251 |

## Class-wise Metrics:

|Class               | Accuracy   |IoU         |Precision  |Recall       |F1 Score    |
|--------------------|------------|------------|-----------|-------------|------------|
|Natural Vegetation  |0.6366      |0.4577      |0.6196     |0.6366       |0.6280      |
|Forest              |0.7171      |0.4772      |0.5878     |0.7171       |0.6461      |
|Corn                |0.6332      |0.5226      |0.7494     |0.6332       |0.6864      |
|Soybeans            |0.6676      |0.51675     |0.6957     |0.6676       |0.6814      |
|Wetlands            |0.6035      |0.4109      |0.5628     |0.6035       |0.5825      |
|Developed/Barren    |0.6022      |0.4637      |0.6684     |0.6022       |0.6336      |
|Open Water          |0.8775      |0.7596      |0.8496     |0.8775       |0.8633      |
|Winter Wheat        |0.6639      |0.4950      |0.6606     |0.6639       |0.6622      |
|Alfalfa             |0.5902      |0.3847      |0.5250     |0.5902       |0.5557      |
|Fallow/Idle Cropland|0.5293      |0.3599      |0.5292     |0.5293       |0.5293      |
|Cotton              |0.4529      |0.3258      |0.5371     |0.4529       |0.4914      |
|Sorghum             |0.6152      |0.3909      |0.5174     |0.6152       |0.5621      |
|Other               |0.4589      |0.3268      |0.5316     |0.4589       |0.4926      |





