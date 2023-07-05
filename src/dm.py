import pytorch_lightning as pl
from pathlib import Path
import pandas as pd
from .utils import get_patch, SPLITS, FOLDS, RARE_FOLDS, AFRICA, AFRICA_2
from .ds import RGBDataset, RGBNirDataset, NonNLDataset, AllDataset, NonNLDatasetCoordCountry, AllDatasetCoordCountry, AllDatasetCoord, NonNLDatasetCoord
from .ds import RGBDatasetCoord, RGBDatasetCoordCountry
from torch.utils.data import DataLoader
import albumentations as A

from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler


class RGBDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, path='data', num_workers=0, folds=AFRICA_2, pin_memory=False, train_trans=None, test_fold=1, val_fold=2, val_size=0.15, random_state=42):
        super().__init__()
        self.batch_size = batch_size
        self.path = Path(path)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_trans = train_trans
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.folds = folds
        self.val_size = val_size
        self.random_state = random_state
        # self.norm_mode = norm_mode

    def read_data(self, mode="train"):
        df = pd.read_csv(self.path / "dhs_final_labels.csv")
        df.dropna(subset=['n_under5_mort'], inplace=True)
        df.rename(columns={'n_under5_mort': 'label'}, inplace=True)
        df['image'] = df['DHSID_EA'].apply(get_patch)
        return df

    def split_data(self):
        self.data_val = self.data[self.data['cname'].isin(
            self.folds[self.val_fold])]
        self.data_test = self.data[self.data['cname'].isin(
            self.folds[self.test_fold])]
        codes = [code for code_list in self.folds.values()
                 for code in code_list]
        self.data_train = self.data[self.data['cname'].isin(codes) & ~self.data['cname'].isin(
            self.folds[self.val_fold] + self.folds[self.test_fold])]

    def split_data_val_size(self):
        self.data_train, self.data_test = train_test_split(
            self.data, test_size=self.val_size, random_state=self.random_state)
        self.data_train, self.data_val = train_test_split(
            self.data_train, test_size=self.val_size, random_state=self.random_state)

    def generate_datasets(self):
        self.ds_train = RGBDataset(
            self.data_train.image.values, self.data_train.label.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ])
            if self.train_trans is not None else None
        )
        self.ds_val = RGBDataset(
            self.data_val.image.values, self.data_val.label.values)
        self.ds_test = RGBDataset(self.data_test.image.values)

    def print_dataset_info(self):
        print('total:', len(self.ds_train) +
              len(self.ds_val) + len(self.ds_test))
        print('train:', len(self.ds_train))
        print('val:', len(self.ds_val))
        print('test:', len(self.ds_test))

    def setup(self, stage=None):
        self.data = self.read_data()
        if self.val_size > 0:
            self.split_data_val_size()
        else:
            self.split_data()
        self.generate_datasets()
        self.print_dataset_info()

    def get_dataloader(self, ds, batch_size=None, shuffle=None):
        return DataLoader(
            ds,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            shuffle=shuffle if shuffle is not None else True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def train_dataloader(self, batch_size=None, shuffle=True):
        return self.get_dataloader(self.ds_train, batch_size, shuffle)

    def val_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.ds_val, batch_size, shuffle)

    def test_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.ds_test, batch_size, shuffle)


class RGBNirDataModule(RGBDataModule):
    def __init__(self, batch_size=32, path='data', num_workers=0, folds=AFRICA_2, pin_memory=False, train_trans=None, test_fold=1, val_fold=2, val_size=0.15, random_state=42):
        super().__init__(batch_size, path, num_workers, folds,
                         pin_memory, train_trans, test_fold, val_fold, val_size, random_state)

    def generate_datasets(self):
        self.ds_train = RGBNirDataset(
            self.data_train.image.values, self.data_train.label.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ])
            if self.train_trans is not None else None
        )
        self.ds_val = RGBNirDataset(
            self.data_val.image.values, self.data_val.label.values)
        self.ds_test = RGBNirDataset(self.data_test.image.values)

    # def setup(self, stage=None):
    #     self.data = self.read_data()
    #     self.split_data()
    #     self.generate_datasets()
    #     self.print_dataset_info()


class NonNLDataModule(RGBDataModule):
    def __init__(self, batch_size=32, path='data', num_workers=0, folds=AFRICA_2, pin_memory=False, train_trans=None, test_fold=1, val_fold=2, val_size=0.15, random_state=42):
        super().__init__(batch_size, path, num_workers, folds,
                         pin_memory, train_trans, test_fold, val_fold, val_size, random_state)

    def generate_datasets(self):
        self.ds_train = NonNLDataset(
            self.data_train.image.values, self.data_train.label.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ])
            if self.train_trans is not None else None
        )
        self.ds_val = NonNLDataset(
            self.data_val.image.values, self.data_val.label.values)
        self.ds_test = NonNLDataset(
            self.data_test.image.values, self.data_test.label.values)

    # def setup(self, stage=None):
    #     self.data = self.read_data()
    #     self.split_data()
    #     self.generate_datasets()
    #     self.print_dataset_info()


class AllDataModule(RGBDataModule):
    def __init__(self, batch_size=32, path='data', num_workers=0, folds=AFRICA_2, pin_memory=False, train_trans=None, test_fold=1, val_fold=2, val_size=0.15, random_state=42):
        super().__init__(batch_size, path, num_workers, folds,
                         pin_memory, train_trans, test_fold, val_fold, val_size, random_state)

    def generate_datasets(self):
        self.ds_train = AllDataset(
            self.data_train.image.values, self.data_train.label.values, self.data_train.year.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ])
            if self.train_trans is not None else None
        )
        self.ds_val = AllDataset(
            self.data_val.image.values, self.data_val.label.values, self.data_val.year.values
        )
        self.ds_test = AllDataset(
            self.data_test.image.values, self.data_test.label.values, self.data_test.year.values)

    # def setup(self, stage=None):
    #     self.data = self.read_data()
    #     self.split_data()
    #     self.generate_datasets()
    #     self.print_dataset_info()


class NonNLDataModuleCoordCountry(RGBDataModule):
    def __init__(self, batch_size=32, path='data', num_workers=0, folds=AFRICA_2, pin_memory=False, train_trans=None, test_fold=1, val_fold=2, val_size=0.15, random_state=42):
        super().__init__(batch_size, path, num_workers, folds,
                         pin_memory, train_trans, test_fold, val_fold, val_size, random_state)

    def generate_datasets(self):
        self.ds_train = NonNLDatasetCoordCountry(
            self.data_train.image.values, self.data_train.label.values, self.data_train.cname.values, self.data_train[['lat', 'lon']].values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ])
            if self.train_trans is not None else None
        )
        self.ds_val = NonNLDatasetCoordCountry(
            self.data_val.image.values, self.data_val.label.values, self.data_val.cname.values,  self.data_val[[
                'lat', 'lon']].values
        )
        self.ds_test = NonNLDatasetCoordCountry(
            self.data_test.image.values, self.data_test.label.values, self.data_test.cname.values,  self.data_test[[
                'lat', 'lon']].values
        )


class AllDataModuleCoordCountry(RGBDataModule):
    def __init__(self, batch_size=32, path='data', num_workers=0, folds=AFRICA_2, pin_memory=False, train_trans=None, test_fold=1, val_fold=2, val_size=0.15, random_state=42):
        super().__init__(batch_size, path, num_workers, folds,
                         pin_memory, train_trans, test_fold, val_fold, val_size, random_state)

    def generate_datasets(self):
        self.ds_train = AllDatasetCoordCountry(
            self.data_train.image.values, self.data_train.label.values, self.data_train.year.values, self.data_train.cname.values, self.data_train[['lat', 'lon']].values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ])
            if self.train_trans is not None else None
        )
        self.ds_val = AllDatasetCoordCountry(
            self.data_val.image.values, self.data_val.label.values, self.data_val.year.values, self.data_val.cname.values, self.data_val[[
                'lat', 'lon']].values
        )
        self.ds_test = AllDatasetCoordCountry(self.data_test.image.values,  self.data_test.label.values, self.data_test.year.values, self.data_test.cname.values, self.data_test[[
            'lat', 'lon']].values
        )


class AllDataModuleCoord(RGBDataModule):
    def __init__(self, batch_size=32, path='data', num_workers=0, folds=AFRICA_2, pin_memory=False, train_trans=None, test_fold=1, val_fold=2, val_size=0.15, random_state=42):
        super().__init__(batch_size, path, num_workers, folds,
                         pin_memory, train_trans, test_fold, val_fold, val_size, random_state)

    def generate_datasets(self):
        self.ds_train = AllDatasetCoord(
            self.data_train.image.values, self.data_train.label.values, self.data_train.year.values, self.data_train[['lat', 'lon']].values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ])
            if self.train_trans is not None else None
        )
        self.ds_val = AllDatasetCoord(
            self.data_val.image.values, self.data_val.label.values, self.data_val.year.values, self.data_val[[
                'lat', 'lon']].values
        )
        self.ds_test = AllDatasetCoord(
            self.data_test.image.values,  self.data_test.label.values, self.data_test.year.values, self.data_test[[
                'lat', 'lon']].values
        )


class NonNLDataModuleCoord(RGBDataModule):
    def __init__(self, batch_size=32, path='data', num_workers=0, folds=AFRICA_2, pin_memory=False, train_trans=None, test_fold=1, val_fold=2, val_size=0.15, random_state=42):
        super().__init__(batch_size, path, num_workers, folds,
                         pin_memory, train_trans, test_fold, val_fold, val_size, random_state)

    def generate_datasets(self):
        self.ds_train = NonNLDatasetCoord(
            self.data_train.image.values, self.data_train.label.values, self.data_train[['lat', 'lon']].values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ])
            if self.train_trans is not None else None
        )
        self.ds_val = NonNLDatasetCoord(
            self.data_val.image.values, self.data_val.label.values, self.data_val[[
                'lat', 'lon']].values
        )
        self.ds_test = NonNLDatasetCoord(
            self.data_test.image.values,  self.data_test.label.values, self.data_test[[
                'lat', 'lon']].values
        )


class RGBDataModuleCoord(RGBDataModule):
    def __init__(self, batch_size=32, path='data', num_workers=0, folds=AFRICA_2, pin_memory=False, train_trans=None, test_fold=1, val_fold=2, val_size=0.15, random_state=42):
        super().__init__(batch_size, path, num_workers, folds,
                         pin_memory, train_trans, test_fold, val_fold, val_size, random_state)

    def generate_datasets(self):
        self.ds_train = RGBDatasetCoord(
            self.data_train.image.values, self.data_train.label.values, self.data_train[['lat', 'lon']].values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ])
            if self.train_trans is not None else None
        )
        self.ds_val = RGBDatasetCoord(
            self.data_val.image.values, self.data_val.label.values, self.data_val[[
                'lat', 'lon']].values
        )
        self.ds_test = RGBDatasetCoord(
            self.data_test.image.values,  self.data_test.label.values, self.data_test[[
                'lat', 'lon']].values
        )


class RGBDataModuleCoordCountry(RGBDataModule):
    def __init__(self, batch_size=32, path='data', num_workers=0, folds=AFRICA_2, pin_memory=False, train_trans=None, test_fold=1, val_fold=2, val_size=0.15, random_state=42):
        super().__init__(batch_size, path, num_workers, folds,
                         pin_memory, train_trans, test_fold, val_fold, val_size, random_state)

    def generate_datasets(self):
        self.ds_train = RGBDatasetCoordCountry(
            self.data_train.image.values, self.data_train.label.values, self.data_train.cname.values, self.data_train[['lat', 'lon']].values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ])
            if self.train_trans is not None else None
        )
        self.ds_val = RGBDatasetCoordCountry(
            self.data_val.image.values, self.data_val.label.values, self.data_val.cname.values,  self.data_val[[
                'lat', 'lon']].values
        )
        self.ds_test = RGBDatasetCoordCountry(
            self.data_test.image.values, self.data_test.label.values, self.data_test.cname.values,  self.data_test[[
                'lat', 'lon']].values
        )
