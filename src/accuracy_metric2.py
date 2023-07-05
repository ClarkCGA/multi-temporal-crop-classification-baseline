import csv
import itertools
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


class Evaluator(object):
    
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)


    def overall_accuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc


    def classwise_overal_accuracy(self):
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return acc


    def precision(self):
        """
        Also known as User’s Accuracy (UA) and Positive Prediction Value (PPV).
        """
        tp = np.diag(self.confusion_matrix)
        fp = np.sum(self.confusion_matrix, axis=0) - tp  # column-wise summation
        precision = tp / (tp + fp)
        return precision


    def recall(self):
        """
        Also known as Producer's Accuracy (PA), True Positive Rate, Sensitivity and hit rate.
        """
        tp = np.diag(self.confusion_matrix)
        fn = np.sum(self.confusion_matrix, axis=1) - tp
        recall = tp / (tp + fn)
        return recall


    def intersection_over_union(self):
        iou = (2 * np.diag(self.confusion_matrix)) / (
                    np.sum(self.confusion_matrix, axis=1) + 
                    np.sum(self.confusion_matrix, axis=0))
        return iou


    def _generate_matrix(self, ref_img, pred_img):
        """
        Generate confusion matrix for a given pair of ground truth and predicted
        images within a batch.

        For each pair in the batch, the resulting confusion matrix is a 2D array 
        where each row corresponds to a class in the ground truth, and each column 
        corresponds to a class in the prediction. The (i, j) element of the matrix 
        is the number of pixels that belong to class i in the ground truth and are 
        classified as class j in the prediction.

        Args:
            ref_img (np.array): 2D array of ref annotation.
            pred_img (np.array): 2D array of model's prediction.

        Returns:
            np.array: A 2D confusion matrix of size (num_class x num_class). 
                      Rows correspond to the true classes and columns correspond 
                      to the predicted classes.
        """
        mask = (ref_img >= 0) & (ref_img < self.num_class)
        label = self.num_class * ref_img[mask].astype('int') + pred_img[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix


    def add_batch(self, ref_img, pred_img):
        """
        update the cumulative confusion matrix with the results from a 
        new batch of images.
        """
        assert ref_img.shape == pred_img.shape
        batch_size = ref_img.shape[0]
        for i in range(batch_size):
            self.confusion_matrix += self._generate_matrix(ref_img[i], 
                                                           pred_img[i])

    def plot_confusion_matrix(self, save_path="confusion_matrix.png"):
        # Remove the first row and column
        conf_mat_without_unknown = self.confusion_matrix[1:, 1:]
        
        # Normalize the confusion matrix by row (i.e., by the true class)
        row_sums = conf_mat_without_unknown.sum(axis=1, keepdims=True)
        conf_mat_normalized = np.divide(conf_mat_without_unknown, row_sums, where=row_sums!=0)
        
        classes = ["Grassland", "Forest", "Shrubs", "Developed", "Open Water", "Woody Wetlands",
                   "Winter Wheats", "Other Hay/Non Alfalfa", "Corn", "Soybeans", "Cottons", "Sorghum", "Other Crops"]
        
        # Create a dataframe for the seaborn heatmap
        df_cm = pd.DataFrame(conf_mat_normalized,
                            index = classes, 
                            columns = classes)
    
        # Create the figure
        plt.figure(figsize=(self.num_class, self.num_class))
    
        # Use seaborn to plot the heatmap
        heatmap = sns.heatmap(df_cm, annot=True, fmt=".3f", cmap='viridis', linewidths=.5, cbar=True)
    
        # Set the title and labels
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted label')
        plt.ylabel('Reference label')
    
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
        plt.show()


    def reset(self):
        """
        Resets the confusion matrix.

        This function sets the confusion matrix back to an empty state, ready to 
        start a new round of evaluation. It can be useful in cases where evaluation 
        is done in an episodic manner, such as when evaluating model performance after
        each epoch during training.
        """
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def do_accuracy_evaluation2(model, dataloader, num_classes, out_name=None):
    evaluator = Evaluator(num_classes)

    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)

            # add batch to evaluator
            evaluator.add_batch(labels.cpu().numpy(), preds.cpu().numpy())

    # calculate evaluation metrics
    overall_accuracy = evaluator.overall_accuracy()
    classwise_overal_accuracy = evaluator.classwise_overal_accuracy()
    mean_accuracy = np.nanmean(classwise_overal_accuracy)
    IoU = evaluator.intersection_over_union()
    mean_IoU = np.nanmean(IoU)
    precision = evaluator.precision()
    mean_precision = np.nanmean(precision)
    recall = evaluator.recall()
    mean_recall = np.nanmean(recall)

    metrics = {
        "Overall Accuracy": overall_accuracy,
        "Mean Accuracy": mean_accuracy,
        "Mean IoU": mean_IoU,
        "mean Precision": mean_precision,
        "mean Recall": mean_recall
    }

    # print confusion matrix
    evaluator.plot_confusion_matrix()
    
    if out_name:
        with open(out_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Metric', 'Value'])

            for metric_name, metric_value in metrics.items():
                writer.writerow([metric_name, metric_value])
        
        class_metrics_out_name = out_name.rsplit('.', 1)[0] + "_classwise." + out_name.rsplit('.', 1)[1]
        with open(class_metrics_out_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Class', 'Accuracy', 'IoU', 'Precision', 'Recall'])

            for i in range(1, evaluator.num_class):
                writer.writerow([i, classwise_overal_accuracy[i], IoU[i], 
                                 precision[i], recall[i]])
    
    return metrics