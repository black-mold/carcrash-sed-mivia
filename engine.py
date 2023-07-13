import pytorch_lightning as pl
import torch

from util import calculate_max_f1score


class FeatureExtractor(pl.LightningModule):
    def __init__(self, preprocess, model, loss_function, optimizer, scheduler):
        super().__init__()

        # ⚡ preprocess
        self.preprocess = preprocess
        print(self.preprocess)

        # ⚡ model
        self.model = model
        print(self.model)

        # ⚡ loss 
        self.loss_function = loss_function

        # ⚡ optimizer
        self.optimizer = optimizer

        # ⚡ scheduler
        self.scheduler = scheduler # **kwargs: **config['scheduler_config']

        # save hyperparameters
        self.save_hyperparameters(ignore=['model'])

    def training_step(self, batch, batch_idx):
        x, y = batch
        # preprocess
        x = self.preprocess(x)

        if x.shape[2] == 7501:
            x = x[:,:,:7500]

        # inference
        y_hat, _ = self.model(x)

        # calculate loss
        loss = self.loss_function(y_hat, y)

        # Logging to TensorBoard
        self.log("loss", loss, on_epoch= True, prog_bar=True, logger=True)

        # f1 score
        f1_score, threshold = calculate_max_f1score(y_hat, y)
        self.log("f1_score_train", f1_score,  on_epoch= True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch

        x = self.preprocess(x)

        if x.shape[2] == 7501:
            x = x[:,:,:7500]
        y_hat, _ = self.model(x)
        loss = self.loss_function(y_hat, y)
        self.log("test_loss", loss,  on_epoch= True, prog_bar=True, logger=True)

        # f1 score
        f1_score, threshold = calculate_max_f1score(y_hat, y)
        self.log("test_f1_score", f1_score,  on_epoch= True, prog_bar=True, logger=True)
        self.log("test_threshold", threshold,  on_epoch= True, prog_bar=True, logger=True)
        

        return {'f1_score': torch.tensor(f1_score) }

    def test_epoch_end(self, outputs):
        avg_f1_score = torch.stack([x['f1_score'] for x in outputs]).mean()
        self.log("avg_test_f1_score", avg_f1_score, on_epoch=True, prog_bar=True, logger=True)



    

    def validation_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch

        x = self.preprocess(x)

        if x.shape[2] == 7501:
            x = x[:,:,:7500]
        y_hat, _ = self.model(x)
        loss = self.loss_function(y_hat, y)
        self.log("validation_loss", loss,  on_epoch= True, prog_bar=True, logger=True)

        # f1 score
        f1_score, threshold = calculate_max_f1score(y_hat, y)
        self.log("validation_f1_score", f1_score,  on_epoch= True, prog_bar=True, logger=True)
        self.log("validation_threshold", threshold,  on_epoch= True, prog_bar=True, logger=True)
        

        return {'f1_score': torch.tensor(f1_score), 'threshold': torch.tensor(threshold)}

    def validation_epoch_end(self, outputs):
        avg_f1_score = torch.stack([x['f1_score'] for x in outputs]).mean()
        avg_threshold = torch.stack([x['threshold'] for x in outputs]).mean()
        self.log("avg_validation_f1_score", avg_f1_score, on_epoch=True, prog_bar=True, logger=True)
        self.log("avg_validation_threshold", avg_threshold, on_epoch=True, prog_bar=True, logger=True)
    


    def forward(self, x):
        y_hat = self.model(x)
        return y_hat


    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "val_loss",
                "frequency": 1
            },
        }