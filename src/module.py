import pytorch_lightning as pl
import timm
import torch.nn.functional as F
import torch
import torch.nn as nn
# from torchmetrics.functional import r2_score
from torchmetrics import PearsonCorrCoef
from .transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotate90, RandomTranspose
from torchmetrics import MeanAbsoluteError
from torchmetrics import R2Score


class RGBModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=1
        )

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            preds = self(x.to(self.device))
            return torch.softmax(preds, dim=1)

    # def shared_step(self, batch, batch_idx):
    #     x, target = batch
    #     y_hat = self(x)
        # predictions = torch.argmax(y_hat, dim=1)

        # cross : pred (N, C), target (N)
        # loss = F.cross_entropy(y_hat, target)

        # r2 = PearsonCorrCoef()(predictions.cpu().float(), target.cpu().float())
        # return loss, r2 ** 2

    def shared_step(self, batch, batch_idx):
        x, target = batch
        y_hat = self(x)

        y_hat = y_hat.squeeze()

        # loss = nn.MSELoss()(y_hat, target)
        # print(y_hat.shape, target.shape)

        mae = MeanAbsoluteError().to(self.device)
        loss = mae(y_hat, target)
        r2 = PearsonCorrCoef()(y_hat.cpu(), target.cpu())
        R2 = R2Score()(y_hat.cpu(), target.cpu())
        # print(loss, r2)
        return loss, r2 ** 2, R2

    def training_step(self, batch, batch_idx):
        loss, r2, R2 = self.shared_step(batch, batch_idx)
        self.log('loss', loss, prog_bar=True, sync_dist=True)
        self.log('r2', r2, prog_bar=True, sync_dist=True)
        self.log('R2', R2, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, r2, R2 = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_r2', r2, prog_bar=True, sync_dist=True)
        self.log('val_R2', R2, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(self.parameters(),
                                                                 **self.hparams['optimizer_params'])
        if 'scheduler' in self.hparams:
            schedulers = [
                getattr(torch.optim.lr_scheduler, scheduler)(
                    optimizer, **params)
                for scheduler, params in self.hparams.scheduler.items()
            ]
            return [optimizer], schedulers
        return optimizer


class RGBNirModule(RGBModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=1,
            in_chans=4,
        )


class NonNLModule(RGBModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=1,
            in_chans=7,
        )


class AllModule(RGBModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=1,
            in_chans=9,
        )


class NonNLModuleCoordCountry(RGBModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=0,
            in_chans=7,
        )

        def layer(h): return nn.Sequential(
            nn.Linear(self.hparams.mlp_layers[h],
                      self.hparams.mlp_layers[h+1]),
            nn.ReLU(),
            nn.Dropout(self.hparams.mlp_dropout)
        )

        self.mlp_coord = nn.Sequential(
            #  coord
            nn.Linear(2, self.hparams.mlp_layers[0]),
            nn.ReLU(),
            nn.Dropout(self.hparams.mlp_dropout),
            *[layer(h) for h in range(len(self.hparams.mlp_layers)-1)],
        )

        self.mlp_country = nn.Sequential(
            #  country
            nn.Linear(56, self.hparams.mlp_layers[0]),
            nn.ReLU(),
            nn.Dropout(self.hparams.mlp_dropout),
            *[layer(h) for h in range(len(self.hparams.mlp_layers)-1)],
        )

        self.classifier = nn.Linear(self.hparams.mlp_layers[-1] +
                                    self.hparams.mlp_layers[-1] + self.model.feature_info[-1]['num_chs'], 1)

    def forward(self, x):
        img, coord, country = x['img'], x['coord'], x['country']
        y_img = self.model(img)
        y_coord = self.mlp_coord(coord)
        country = F.one_hot(torch.tensor(country), 56).float()
        y_country = self.mlp_country(country)
        f = torch.cat((y_country, y_coord, y_img.squeeze()), dim=-1)
        return self.classifier(f)

    def shared_step(self, batch, batch_idx):
        target = batch['label']
        y_hat = self(batch)

        y_hat = y_hat.squeeze()

        # loss = nn.MSELoss()(y_hat, target)
        # print(y_hat.shape, target.shape)
        mae = MeanAbsoluteError().to(self.device)
        loss = mae(y_hat, target)
        r2 = PearsonCorrCoef()(y_hat.cpu(), target.cpu())
        R2 = R2Score()(y_hat.cpu(), target.cpu())
        # print(loss, r2)
        return loss, r2 ** 2, R2


class AllModuleCoordCountry(NonNLModuleCoordCountry):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=0,
            in_chans=9,
        )

        def layer(h): return nn.Sequential(
            nn.Linear(self.hparams.mlp_layers[h],
                      self.hparams.mlp_layers[h+1]),
            nn.ReLU(),
            nn.Dropout(self.hparams.mlp_dropout)
        )

        self.mlp_coord = nn.Sequential(
            #  coord
            nn.Linear(2, self.hparams.mlp_layers[0]),
            nn.ReLU(),
            nn.Dropout(self.hparams.mlp_dropout),
            *[layer(h) for h in range(len(self.hparams.mlp_layers)-1)],
        )

        self.mlp_country = nn.Sequential(
            #  country
            nn.Linear(56, self.hparams.mlp_layers[0]),
            nn.ReLU(),
            nn.Dropout(self.hparams.mlp_dropout),
            *[layer(h) for h in range(len(self.hparams.mlp_layers)-1)],
        )

        self.classifier = nn.Linear(self.hparams.mlp_layers[-1] +
                                    self.hparams.mlp_layers[-1] + self.model.feature_info[-1]['num_chs'], 1)


class NonNLModuleCoord(RGBModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=0,
            in_chans=7,
        )

        def layer(h): return nn.Sequential(
            nn.Linear(self.hparams.mlp_layers[h],
                      self.hparams.mlp_layers[h+1]),
            nn.ReLU(),
            nn.Dropout(self.hparams.mlp_dropout)
        )

        self.mlp_coord = nn.Sequential(
            #  coord
            nn.Linear(2, self.hparams.mlp_layers[0]),
            nn.ReLU(),
            nn.Dropout(self.hparams.mlp_dropout),
            *[layer(h) for h in range(len(self.hparams.mlp_layers)-1)],
        )

        self.classifier = nn.Linear(
            self.hparams.mlp_layers[-1] + self.model.feature_info[-1]['num_chs'], 1)

    def forward(self, x):
        img, coord = x['img'], x['coord']
        y_img = self.model(img)
        y_coord = self.mlp_coord(coord)
        f = torch.cat((y_coord, y_img.squeeze()), dim=-1)
        return self.classifier(f)

    def shared_step(self, batch, batch_idx):
        target = batch['label']
        y_hat = self(batch)

        y_hat = y_hat.squeeze()

        # loss = nn.MSELoss()(y_hat, target)
        # print(y_hat.shape, target.shape)
        mae = MeanAbsoluteError().to(self.device)
        loss = mae(y_hat, target)
        r2 = PearsonCorrCoef()(y_hat.cpu(), target.cpu())
        R2 = R2Score()(y_hat.cpu(), target.cpu())
        # print(loss, r2)
        return loss, r2 ** 2, R2


class AllModuleCoord(NonNLModuleCoord):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=0,
            in_chans=9,
        )

        def layer(h): return nn.Sequential(
            nn.Linear(self.hparams.mlp_layers[h],
                      self.hparams.mlp_layers[h+1]),
            nn.ReLU(),
            nn.Dropout(self.hparams.mlp_dropout)
        )

        self.mlp_coord = nn.Sequential(
            #  coord
            nn.Linear(2, self.hparams.mlp_layers[0]),
            nn.ReLU(),
            nn.Dropout(self.hparams.mlp_dropout),
            *[layer(h) for h in range(len(self.hparams.mlp_layers)-1)],
        )

        self.classifier = nn.Linear(
            self.hparams.mlp_layers[-1] + self.model.feature_info[-1]['num_chs'], 1)


class RGBModuleCoord(NonNLModuleCoord):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=0,
            in_chans=3,
        )

        def layer(h): return nn.Sequential(
            nn.Linear(self.hparams.mlp_layers[h],
                      self.hparams.mlp_layers[h+1]),
            nn.ReLU(),
            nn.Dropout(self.hparams.mlp_dropout)
        )

        self.mlp_coord = nn.Sequential(
            #  coord
            nn.Linear(2, self.hparams.mlp_layers[0]),
            nn.ReLU(),
            nn.Dropout(self.hparams.mlp_dropout),
            *[layer(h) for h in range(len(self.hparams.mlp_layers)-1)],
        )

        self.classifier = nn.Linear(
            self.hparams.mlp_layers[-1] + self.model.feature_info[-1]['num_chs'], 1)


class RGBModuleCoordCountry(NonNLModuleCoordCountry):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=0,
            in_chans=3,
        )

        def layer(h): return nn.Sequential(
            nn.Linear(self.hparams.mlp_layers[h],
                      self.hparams.mlp_layers[h+1]),
            nn.ReLU(),
            nn.Dropout(self.hparams.mlp_dropout)
        )

        self.mlp_coord = nn.Sequential(
            #  coord
            nn.Linear(2, self.hparams.mlp_layers[0]),
            nn.ReLU(),
            nn.Dropout(self.hparams.mlp_dropout),
            *[layer(h) for h in range(len(self.hparams.mlp_layers)-1)],
        )

        self.mlp_country = nn.Sequential(
            #  country
            nn.Linear(56, self.hparams.mlp_layers[0]),
            nn.ReLU(),
            nn.Dropout(self.hparams.mlp_dropout),
            *[layer(h) for h in range(len(self.hparams.mlp_layers)-1)],
        )

        self.classifier = nn.Linear(self.hparams.mlp_layers[-1] +
                                    self.hparams.mlp_layers[-1] + self.model.feature_info[-1]['num_chs'], 1)


class NonNLAndNLModuleCoordCountry(RGBModule):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.modelNonNL = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=0,
            in_chans=7,
        )
        self.modelNL = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=0,
            in_chans=2,
        )

        def layer(h): return nn.Sequential(
            nn.Linear(self.hparams.mlp_layers[h],
                      self.hparams.mlp_layers[h+1]),
            nn.ReLU(),
            nn.Dropout(self.hparams.mlp_dropout)
        )

        self.mlp_coord = nn.Sequential(
            #  coord
            nn.Linear(2, self.hparams.mlp_layers[0]),
            nn.ReLU(),
            nn.Dropout(self.hparams.mlp_dropout),
            *[layer(h) for h in range(len(self.hparams.mlp_layers)-1)],
        )

        self.mlp_country = nn.Sequential(
            #  country
            nn.Linear(56, self.hparams.mlp_layers[0]),
            nn.ReLU(),
            nn.Dropout(self.hparams.mlp_dropout),
            *[layer(h) for h in range(len(self.hparams.mlp_layers)-1)],
        )

        self.classifier = nn.Linear(self.hparams.mlp_layers[-1] +
                                    self.hparams.mlp_layers[-1] + 2 * self.model.feature_info[-1]['num_chs'], 1)

    def forward(self, x):
        # print(x['img'][:, :7].shape, x['img'][:, 7:].shape)
        imgNonNL, imgNL, coord, country = x['img'][:, :7], x['img'][:, 7:], x['coord'], x['country']
        y_imgNonNL = self.modelNonNL(imgNonNL)
        y_imgNL = self.modelNL(imgNL)
        y_coord = self.mlp_coord(coord)
        country = F.one_hot(torch.tensor(country), 56).float()
        y_country = self.mlp_country(country)
        f = torch.cat((y_country, y_coord, y_imgNonNL.squeeze(),
                      y_imgNL.squeeze()), dim=-1)
        return self.classifier(f)

    def shared_step(self, batch, batch_idx):
        target = batch['label']
        y_hat = self(batch)

        y_hat = y_hat.squeeze()

        mae = MeanAbsoluteError().to(self.device)
        loss = mae(y_hat, target)
        r2 = PearsonCorrCoef()(y_hat.cpu(), target.cpu())
        R2 = R2Score()(y_hat.cpu(), target.cpu())

        return loss, r2 ** 2, R2
