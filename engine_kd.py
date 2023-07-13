import pytorch_lightning as pl
import torch

from util import calculate_max_f1score


class FeatureExtractor(pl.LightningModule):
    def __init__(self, preprocess, model_teacher, model_student, loss_function_bce, loss_function_mse, optimizer, scheduler):
        super().__init__()

        # ⚡ preprocess
        self.preprocess = preprocess
        print(self.preprocess)

        # ⚡ model
        self.model_teacher = model_teacher
        print(self.model_teacher)

        self.model_student = model_student
        print(self.model_student)

        # ⚡ loss 
        self.loss_function_bce = loss_function_bce
        self.loss_function_mse = loss_function_mse

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
        self.model_teacher.eval()
        with torch.no_grad():
            y_teacher, _ = self.model_teacher(x)
        y_student, _ = self.model_student(x)

        # calculate loss
        loss_bce = self.loss_function_bce(y_student, y)

        loss_mse = self.loss_function_mse(y_teacher, y_student)

        loss = (loss_bce + loss_mse)/2

        # Logging to TensorBoard
        self.log("loss_bce", loss_bce, on_epoch= True, prog_bar=True, logger=True)
        self.log("loss_mse", loss_mse, on_epoch= True, prog_bar=True, logger=True)
        self.log("loss", loss, on_epoch= True, prog_bar=True, logger=True)

        # f1 score
        f1_score, threshold = calculate_max_f1score(y_student, y)
        self.log("f1_score_train", f1_score,  on_epoch= True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch

        x = self.preprocess(x)

        if x.shape[2] == 7501:
            x = x[:,:,:7500]
        y_student, _ = self.model_student(x)
        loss_bce = self.loss_function_bce(y_student, y)
        self.log("test_loss_bce", loss_bce,  on_epoch= True, prog_bar=True, logger=True)

        # f1 score
        f1_score, threshold = calculate_max_f1score(y_student, y)
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
        y_student, _ = self.model_student(x)
        loss_bce = self.loss_function_bce(y_student, y)
        self.log("validation_loss", loss_bce,  on_epoch= True, prog_bar=True, logger=True)

        # f1 score
        f1_score, threshold = calculate_max_f1score(y_student, y)
        self.log("validation_f1_score", f1_score,  on_epoch= True, prog_bar=True, logger=True)
        self.log("validation_threshold", threshold,  on_epoch= True, prog_bar=True, logger=True)
        

        return {'f1_score': torch.tensor(f1_score), 'threshold': torch.tensor(threshold)}

    def validation_epoch_end(self, outputs):
        avg_f1_score = torch.stack([x['f1_score'] for x in outputs]).mean()
        avg_threshold = torch.stack([x['threshold'] for x in outputs]).mean()
        self.log("avg_validation_f1_score", avg_f1_score, on_epoch=True, prog_bar=True, logger=True)
        self.log("avg_validation_threshold", avg_threshold, on_epoch=True, prog_bar=True, logger=True)
    


    def forward(self, x):
        y_hat = self.model_student(x)
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