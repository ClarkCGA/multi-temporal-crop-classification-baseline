import csv
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def pixel_accuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def pixel_accuracy_class(self):
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        #acc = np.nanmean(acc)
        if np.isnan(acc).any():
            print('Warning: NaN values encountered in pixel_accuracy_class computation.')
            acc = acc[~np.isnan(acc)]
        # compute mean if array is not empty, else assign 0
        acc = np.mean(acc) if acc.size else 0
        return acc

    def mean_intersection_over_union(self):
        miou = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        #miou = np.nanmean(miou)
        if np.isnan(miou).any():
            print('Warning: NaN values encountered in mean_intersection_over_union computation')
            miou = miou[~np.isnan(miou)]
        miou = np.mean(miou) if miou.size else 0
        return miou

    def frequency_weighted_intersection_over_union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        fwiou = (freq[freq > 0] * iu[freq > 0]).sum()
        return fwiou

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
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
    pixel_accuracy = evaluator.Pixel_Accuracy()
    mean_accuracy = evaluator.Pixel_Accuracy_Class()
    mean_IoU = evaluator.Mean_Intersection_over_Union()
    frequency_weighted_IoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    metrics = {
        'Pixel Accuracy': pixel_accuracy,
        'Mean Accuracy': mean_accuracy,
        'Mean IoU': mean_IoU,
        'Frequency Weighted IoU': frequency_weighted_IoU
    }

    if out_name:
        with open(out_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Metric', 'Value'])

            for metric_name, metric_value in metrics.items():
                writer.writerow([metric_name, metric_value])
    
    return metrics