import torch
import torch.nn as nn
import pytorch_lightning as pl

from ttbb_dctr.models.binary_classifier import BinaryClassifier

class BinaryClassifierWithThreshold(BinaryClassifier):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers, learning_rate=1e-3, weight_decay=0, score_threshold=0.1):
        super().__init__(input_size, hidden_size, output_size, num_hidden_layers, learning_rate, weight_decay)
        self.score_threshold = score_threshold
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        y_pred = self(x)
        mask = y_pred >= self.score_threshold
        losses = self.criterion(y_pred[mask].squeeze(), y[mask].float())
        loss = (losses * w[mask]).mean()
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        y_pred = self(x)
        mask = y_pred >= self.score_threshold
        losses = self.criterion(y_pred[mask].squeeze(), y[mask].float())
        loss = (losses * w[mask]).mean()
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, w = batch
        y_pred = self(x)
        mask = y_pred >= self.score_threshold
        losses = self.criterion(y_pred[mask].squeeze(), y[mask].float())
        loss = (losses * w[mask]).mean()
        self.log('test_loss', loss, prog_bar=True)
        return loss