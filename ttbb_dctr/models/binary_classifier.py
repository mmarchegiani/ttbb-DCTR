import torch
import torch.nn as nn
import pytorch_lightning as pl

class BinaryClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers, learning_rate=1e-3, weight_decay=0):
        super(BinaryClassifier, self).__init__()
        self.save_hyperparameters()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none') # Note: with the BCEWithLogitsLoss, a sigmoid is applied to the output of the model

        layers = [
            nn.BatchNorm1d(input_size, affine=False), # Batch normalization layer
            nn.Linear(input_size, hidden_size) # Input layer
        ]
        
        # Stack N-1 hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, hidden_size)) # Hidden layers

        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size)) # Output layer

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        
        return x

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        y_pred = self(x)
        losses = self.criterion(y_pred.squeeze(), y.float())
        loss = (losses * w).mean()
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        y_pred = self(x)
        losses = self.criterion(y_pred.squeeze(), y.float())
        loss = (losses * w).mean()
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, w = batch
        y_pred = self(x)
        losses = self.criterion(y_pred.squeeze(), y.float())
        loss = (losses * w).mean()
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y, w = batch
        y_pred = self(x)
        y_pred = torch.sigmoid(y_pred)
        return y_pred

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
