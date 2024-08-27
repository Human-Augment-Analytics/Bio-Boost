"""
Results, Class Imbalance

The following script was used to calculate various metrics used in the results.
"""

# Encoding, etc.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import numpy as np

# Metrics Calculation
class Metrics:
    """
    A class to calculate various metrics for binary classification.

    Attributes:
        label (array-like): Ground truth binary labels.
        predict (array-like): Predicted binary labels.
        tp (int): True Positives.
        tn (int): True Negatives.
        fp (int): False Positives.
        fn (int): False Negatives.
        total (int): Total number of samples.
        pp (float): Proportion of positive samples.
        pn (float): Proportion of negative samples.
        tpr (float): True Positive Rate (Sensitivity).
        tnr (float): True Negative Rate (Specificity).
        fpr (float): False Positive Rate.
        fnr (float): False Negative Rate.
        acc (float): Accuracy.
        pprecision (float): Positive Precision.
        nprecision (float): Negative Precision.
        precall (float): Positive Recall.
        nrecall (float): Negative Recall.
        pFm (float): Positive F-measure.
        nFm (float): Negative F-measure.
        bacc (float): Balanced Accuracy.
        mcc (float): Matthews Correlation Coefficient.

    Example Usage:
        metrics = Metrics(label, predict)
        print("True Positives:", metrics.tp)
    """
    def __init__(self, label, predict):
        """
        Initialize the Metrics class with ground truth labels and predictions.

        Parameters:
            label (array-like): Ground truth binary labels.
            predict (array-like): Predicted binary labels.
        """
        print('label=1 is positive, label=0 is negative')
        self.label=label
        self.predict=predict

        self.tp = np.sum((self.label == 1) & (self.predict == 1))
        self.tn = np.sum((self.label == 0) & (self.predict == 0))
        self.fp = np.sum((self.label == 0) & (self.predict == 1))
        self.fn = np.sum((self.label == 1) & (self.predict == 0))

        # Rates
        self.total=self.tp+self.tn+self.fp+self.fn
        self.pp=(self.tp+self.fn)/self.total
        self.pn=(self.tn+self.fp)/self.total
        self.tpr=self.tp/(self.tp+self.fn)
        self.tnr=self.tn/(self.tn+self.fp)
        self.fpr=self.fp/(self.tn+self.fp)
        self.fnr=self.fn/(self.tp+self.fn)

        # General Metrics
        self.acc=(self.tp+self.tn)/self.total
        self.pprecision=self.tp/(self.tp+self.fp)
        self.nprecision=self.tn/(self.tn+self.fn)
        self.precall=self.tpr
        self.nrecall=self.tnr
        self.pFm=2*(self.tp)/(2*self.tp+self.tn+self.fp)
        self.nFm=2*(self.tn)/(2*self.tn+self.tp+self.fn)
        
        # For Unbalanced Datasets
        self.bacc=0.5*(self.tpr+self.tnr)
        self.mcc=(self.tp*self.tn-self.fp*self.fn)/np.sqrt((self.tp+self.fp)*(self.tp+self.fn)*(self.tn+self.fp)*(self.tn+self.fn))
        