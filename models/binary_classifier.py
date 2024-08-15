import pytorch_lightning as pl
import torch.nn as nn

class BinaryClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers, learning_rate=1e-3):
        super(BinaryClassifier, self).__init__()
        self.learning_rate = learning_rate
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

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

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        y_pred = self(x)
        losses = self.criterion(y_pred.squeeze(), y.float())
        loss = (losses * w).mean()
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, w = batch
        y_pred = self(x)
        losses = self.criterion(y_pred.squeeze(), y.float())
        loss = (losses * w).mean()
        #acc = self.accuracy(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        #self.log('test_acc', acc, prog_bar=True)
        #return {'loss': loss, 'acc': acc}
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
