import torch, torchmetrics
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchxrayvision as xrv


class SiameseModel(pl.LightningModule):
    def __init__(self,  config):
        super(SiameseModel, self).__init__()
        self.nclasses = config['nclasses']
        self.cnn1 = xrv.autoencoders.ResNetAE(weights="101-elastic")
        outdim = 512 * 3 * 3 * 2
        if config['freeze']: self.freeze_layers()
        self.fc = nn.Linear(outdim, config['nnsize'])
        self.dropout = nn.Dropout(config['dropout'], inplace=True)
        self.fc_final = nn.Linear(config['nnsize'], self.nclasses)
        self.learning_rate = config['lr']

    def freeze_layers(self):
        for param in self.cnn1.parameters():
            param.requires_grad = False

    def forward_once(self, x):
        return self.cnn1(x)["z"].view(-1, 512*3*3)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = torch.cat((output1, output2),1)
        output = F.relu(self.fc(output))
        output = self.dropout(output)
        output = self.fc_final(output)
        return output

    def training_step(self, batch, batch_idx):
        img0, img1, label, meta = batch
        preds = self(img0, img1)
        loss = F.cross_entropy(preds, label)
        self.log('train_loss', loss)
        self.log("train_acc", torchmetrics.functional.accuracy(torch.softmax(preds, dim=-1), label),  prog_bar=True, logger=True)
        self.log("train_f1", torchmetrics.functional.f1(torch.softmax(preds, dim=-1), label, average='macro', num_classes=self.nclasses), prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img0, img1, label, meta = batch
        preds = self(img0, img1)
        loss = F.cross_entropy(preds, label)
        accuracy = torchmetrics.functional.accuracy(torch.softmax(preds, dim=-1), label)
        f1 = torchmetrics.functional.f1(torch.softmax(preds, dim=-1), label, average='macro', num_classes=self.nclasses)
        return {"val_loss": loss, "val_acc": accuracy, "val_f1": f1}

    def validation_epoch_end(self, batch):
        avg_loss = torch.stack([x["val_loss"] for x in batch]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in batch]).mean()
        avg_f1 = torch.stack([x["val_f1"] for x in batch]).mean()
        print(f'\nVal acc: {avg_acc}, Val F1:{avg_f1}\n')
        self.log("val_loss", avg_loss,   prog_bar=True, logger=True)
        self.log("val_acc", avg_acc,   prog_bar=True, logger=True)
        self.log("val_f1", avg_f1,  prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        img0, img1, label, meta = batch
        preds = self(img0, img1)
        loss = F.cross_entropy(preds, label)
        self.log('test_loss', loss)
        self.log("test_acc", torchmetrics.functional.accuracy(torch.softmax(preds, dim=-1), label), prog_bar=True, logger=True)
        self.log("test_f1", torchmetrics.functional.f1(torch.softmax(preds, dim=-1), label, average='macro', num_classes=self.nclasses), prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)




