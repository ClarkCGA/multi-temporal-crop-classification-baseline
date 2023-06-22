import numpy as np
import pandas as pd
from torch.autograd import Variable
import torch.nn.functional as F


class BinaryMetrics:
    """
    Metrics measuring model performance.
    """

    def __init__(self, ref_array, score_array, pred_array=None):
        """
        Params:
            ref_array (ndarray): Array of ground truth
            score_array (ndarray): Array of pixels scores of positive class
            pred_array (ndarray): Boolean array of predictions telling whether
                                 a pixel belongs to a specific class.
        """

        self.tp = None
        self.fp = None
        self.fn = None
        self.tn = None
        self.eps = 10e-6
        self.observation = ref_array.flatten()
        self.score = score_array.flatten()
        if pred_array is not None:
            self.prediction = pred_array.flatten()
        # take score over 0.5 as prediction if predArray not provided
        else:
            self.prediction = np.where(self.score > 0.5, 1, 0)
        self.confusion_matrix = self.confusion_matrix()

        assert self.observation.shape == self.score.shape, "Inconsistent input shapes"

    def __add__(self, other):
        """
        Add two BinaryMetrics instances
        Params:
            other (''BinaryMetrics''): A BinaryMetrics instance
        Return:
            ''BinaryMetrics''
        """

        return BinaryMetrics(np.append(self.observation, other.observation),
                             np.append(self.score, other.score),
                             np.append(self.prediction, other.prediction))

    def __radd__(self, other):
        """
        Add a BinaryMetrics instance with reversed operands
        Params:
            other
        Returns:
            ''BinaryMetrics
        """

        if other == 0:
            return self
        else:
            return self.__add__(other)

    def confusion_matrix(self):
        """
        Calculate confusion matrix of given ground truth and predicted label
        Returns:
            "pandas.dataframe" of observation on the column and prediction on the row
        """

        ref_array = self.observation
        pred_array = self.prediction

        if ref_array.max() > 1 or pred_array.max() > 1:
            raise Exception("Invalid array")
        predArray = pred_array * 2
        sub = ref_array - predArray

        self.tp = np.sum(sub == -1)
        self.fp = np.sum(sub == -2)
        self.fn = np.sum(sub == 1)
        self.tn = np.sum(sub == 0)

        confusionMatrix = pd.DataFrame(data=np.array([[self.tn, self.fp], [self.fn, self.tp]]),
                                       index=['observation = 0', 'observation = 1'],
                                       columns=['prediction = 0', 'prediction = 1'])
        return confusionMatrix

    
    def ir(self):
        """
        Imbalance Ratio (IR) is defined as the proportion between positive and negative
        instances of the label. This value lies within the [0, ∞] range, having a value
        IR = 1 in the balanced case.
        Returns:
                float
        """     
        ir = np.divide(self.tp + self.fn, self.tn + self.fp)

        if np.isinf(ir) or np.isnan(ir):
            ir = np.divide(self.tp + self.fn, self.fp + self.tn + self.eps)

        return ir

    
    def accuracy(self):
        """
        Calculate Overall (Global) Accuracy.
        Returns:
            float scalar
        """       
        oa = np.divide(self.tp + self.tn, self.tp + self.tn + self.fp + self.fn)

        if np.isinf(oa) or np.isnan(oa):
            oa = np.divide(self.tp + self.tn, self.tp + self.tn + self.fp + self.fn + self.eps)

        return oa
    
    def precision(self):
        """
        Calculate User’s Accuracy (Positive Prediction Value (PPV) | UA).
        Returns:
            float
        """
        ua = np.divide(self.tp, self.tp + self.fp, where=(self.tp+self.fp)!=0)
        invalid_values_mask = np.isnan(ua) | np.isinf(ua)
    
        if np.any(invalid_values_mask):
            print(f"Invalid values encountered in precision computation: {ua[invalid_values_mask]}")

        ua = np.where((self.tp+self.fp)!=0, ua, np.divide(self.tp, 
                                                          self.tp + self.fp + self.eps, 
                                                          where=(self.tp+self.fp+self.eps)!=0))

        invalid_values_mask = np.isnan(ua) | np.isinf(ua)
        if np.any(invalid_values_mask):
            print(f"Invalid values encountered in precision computation after applying epsilon: {ua[invalid_values_mask]}")

        return ua
    

    def recall(self):
        """
        Calculate Producer's Accuracy (True Positive Rate |Sensitivity |hit rate | recall).
        Returns:
            float
        """

        pa = np.divide(self.tp, self.tp + self.fn, where=(self.tp+self.fn)!=0)
        invalid_values_mask = np.isnan(pa) | np.isinf(pa)
    
        if np.any(invalid_values_mask):
            print(f"Invalid values encountered in recall computation: {pa[invalid_values_mask]}")

        pa = np.where((self.tp+self.fn)!=0, pa, np.divide(self.tp, self.tp + self.fn + self.eps, where=(self.tp+self.fn+self.eps)!=0))
    
        invalid_values_mask = np.isnan(pa) | np.isinf(pa)
        if np.any(invalid_values_mask):
            print(f"Invalid values encountered in recall computation after applying epsilon: {pa[invalid_values_mask]}")

        return pa

    def false_positive_rate(self):
        """
        Calculate False Positive Rate(FPR) aka. False Alarm Rate (FAR), or Fallout.
        Returns:
             float
        """
        
        fpr = np.divide(self.tp, self.tn + self.fp)

        if np.isinf(fpr) or np.isnan(fpr):
            ua = np.divide(self.fp, self.tn + self.fp + self.eps)

        return fpr

    def iou(self):
        """
        Calculate interception over union for the positive class.
        Returns:
            float
        """
        
        iou = np.divide(self.tp, self.tp + self.fp + self.fn)

        if np.isinf(iou) or np.isnan(iou):
            iou = np.divide(self.tp, self.tp + self.fp + self.fn + self.eps)

        return iou

    def f1_measure(self):
        """
        Calculate F1 score.
        Returns:
            float
        """
        precision = np.divide(self.tp, self.tp + self.fp, where=(self.tp+self.fp)!=0)
        invalid_values_mask = np.isnan(precision) | np.isinf(precision)
        if np.any(invalid_values_mask):
            print(f"Invalid values encountered in precision computation: {precision[invalid_values_mask]}")
        precision = np.where((self.tp+self.fp)!=0, 
                             precision, 
                             np.divide(self.tp, self.tp + self.fp + self.eps, where=(self.tp+self.fp+self.eps)!=0))

        recall = np.divide(self.tp, self.tp + self.fn, where=(self.tp+self.fn)!=0)
        invalid_values_mask = np.isnan(recall) | np.isinf(recall)
        if np.any(invalid_values_mask):
            print(f"Invalid values encountered in recall computation: {recall[invalid_values_mask]}")
        recall = np.where((self.tp+self.fn)!=0, 
                          recall, 
                          np.divide(self.tp, self.tp + self.fn + self.eps, where=(self.tp+self.fn+self.eps)!=0))

        f1 = np.divide(2 * precision * recall, precision + recall, where=(precision+recall)!=0)
        invalid_values_mask = np.isnan(f1) | np.isinf(f1)
        if np.any(invalid_values_mask):
            print(f"Invalid values encountered in F1 computation: {f1[invalid_values_mask]}")
        f1 = np.where((precision+recall)!=0, f1, np.divide(2 * precision * recall, precision + recall + self.eps, where=(precision+recall+self.eps)!=0))

        return f1

    def tss(self):
        """
        Calculate true skill statistic (TSS)
        Returns:
            float
        """
        tp_rate = np.divide(self.tp, self.tp + self.fn)
        if np.isinf(tp_rate) or np.isnan(tp_rate):
            tp_rate = np.divide(self.tp, self.tp + self.fn + self.eps)

        tn_rate = np.divide(self.tn, self.tn + self.fp)
        if np.isinf(tn_rate) or np.isnan(tn_rate):
            tn_rate = np.divide(self.tn, self.tn + self.fp + self.eps)

        return tp_rate + tn_rate - 1


def do_accuracy_evaluation(eval_data, model, filename, gpu=True):
    r"""
    Evaluate the model on a separate Landsat scene.

    Arguments:
    eval_data -- Batches of image chips from PyTorch custom dataset(AquacultureData)
    model -- Choice of segmentation Model to train.
    filename -- (str) Name of the csv file to report metrics.
    gpu --(binary) If False the model will run on CPU instead of GPU. Default is True.

    Note: to harden the class prediction around a higher probability, drop 'class_pred' argument
          and increase the threshold of 'predArray' in the 'BinaryMetrics' class '__init__' function.

    """

    model.eval()

    metrics_ls = []

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for img_chips, label in eval_data:

        img = Variable(img_chips, requires_grad=False)  # size: batch size X channels X W X H
        label = Variable(label, requires_grad=False)  # size: batch size X W X H

        if gpu:
            img = img.cuda()
            label = label.cuda()

        pred = model(img)  # size: batch size x number of categories X W x H
        pred_prob = F.softmax(pred, 1)
        batch, n_class, height, width = pred_prob.size()

        for i in range(batch):
            label_batch = label[i, :, :].cpu().numpy()
            batch_pred = pred_prob.max(dim=1)[1][:, :, :].data[i].cpu().numpy()

            for n in range(1, n_class):
                class_prob = pred_prob[:, n, :, :].data[i].cpu().numpy()
                class_pred = np.where(batch_pred == n, 1, 0)
                class_label = np.where(label_batch == n, 1, 0)
                chip_metrics = BinaryMetrics(class_label, class_prob, class_pred)

                try:
                    metrics_ls[n - 1].append(chip_metrics)
                except:
                    metrics_ls.append([chip_metrics])

    metrics = [sum(m) for m in metrics_ls]

    report = pd.DataFrame({
        "Imbalance Ratio": [m.ir() for m in metrics],
        "Overall Accuracy": [m.accuracy() for m in metrics],
        "Precision (UA or PPV)": [m.precision() for m in metrics],
        "Recall (PA or TPR or Sensitivity)": [m.recall() for m in metrics],
        "False Positive Rate": [m.false_positive_rate() for m in metrics],
        "IoU": [m.iou() for m in metrics],
        "F1-score": [m.f1_measure() for m in metrics],
        "TSS": [m.tss() for m in metrics]
    }, ["class_{}".format(m) for m in range(1, len(metrics) + 1)])

    report.to_csv(filename, index=False)