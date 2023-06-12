import os
import torch
import rasterio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
import torch.nn.functional as F


def do_prediction(testData, model, score_path, prob_path, gpu):
    r"""
    Use train model to predict on unseen data.

    Arguments:
        testData (custom iterator): Batches of the tuple (image chips, img_ids, img_meta) 
            from PyTorch custom dataset.
                'img_ids' (list of strings)-- tile identifier for each image tile in the batch
                     in the form of 6 digits seperated in the middle with an underscore like "003_012"
                'img_meta' (list of dictionary)-- Rasterio metadata for each image chip in the batch 
        model (ordered Dict): trained model.
        score_path (str): Directory to store hardened probability map.
        prob_path (str): Directory to store soft probability map.
        gpu (binary): If False the model will run on CPU instead of GPU. Default is True.
    output:
        hardened prediction as tiff using rasterio in score_path. Filename should 
        be "score_{img_id}" and use the 'img_meta'.
          
    """
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("Warning: Prediction is running on CPU.")
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    with torch.no_grad():

        for images, img_ids, img_metas in testData:
            
            img_chips = images.to(device)
            outputs = model(img_chips)
            soft_preds = F.softmax(outputs, 1)
            hard_preds = soft_preds.max(dim=1)

            batch, n_class, height, width = soft_preds.size()

            assert len(img_ids) == batch == len(img_metas)

            for i in range(batch):
                
                img_id = img_ids[i]
                name_prob = f"prob_id_{img_id}"
                name_crisp = f"crisp_id_{img_id}.tif"
                
                img_meta = img_metas[i]
                
                meta_hard = img_meta.copy()
                meta_hard.update({
                    "dtype": "uint8",
                    "count": 1,
                    })
                
                meta_soft = img_meta.copy()
                meta_soft.update({
                    "dtype": "float32",
                    "count": 1,
                    })
                
                hard_pred = hard_preds[i].cpu().numpy().astype(meta_hard["dtype"])
                with rasterio.open(Path(score_path) / name_crisp, "w", **meta_hard) as dst:
                    dst.write(hard_pred, 1)
                
                for n in range(1, n_class):
                    name_prob_updated = f"{name_prob}_cat_{n}.tif"
                    soft_pred = soft_preds[:, n, :, :].data[i].cpu().numpy( ) * 100
                    soft_pred = np.expand_dims(soft_pred, axis=0).astype(meta_soft["dtype"])

                    with rasterio.open(Path(prob_path) / name_prob_updated, "w", **meta_soft) as dst:
                        dst.write(soft_pred, 1)
                
 