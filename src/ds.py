import torch
# from .utils import get_patch, get_country
import numpy as np
import pandas as pd
from .utils import COUNTRYS


class RGBDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, trans=None):
        self.images = images
        self.labels = labels
        self.trans = trans

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        img = np.load(self.images[ix])['x'][0:3]
        label = self.labels[ix]
        # label = int(label)
        if self.trans is not None:
            img = np.transpose(img, (1, 2, 0))
            img = self.trans(image=img)['image']
            img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img),  torch.tensor(label).float()


class RGBNirDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, trans=None):
        self.images = images
        self.labels = labels
        self.trans = trans

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        img = np.load(self.images[ix])['x'][[0, 1, 2, 6]]
        label = self.labels[ix]
        # label = int(label)
        if self.trans is not None:
            img = np.transpose(img, (1, 2, 0))
            img = self.trans(image=img)['image']
            img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img),  torch.tensor(label).float()


class NonNLDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, path='data', stats_path='real_stats.csv', norm_mode='min_max', trans=None):
        self.images = images
        self.labels = labels
        self.trans = trans
        self.path = path
        self.norm_mode = norm_mode
        self.stats = pd.read_csv(f'{self.path}/{stats_path}', index_col=0)

    def __len__(self):
        return len(self.images)

    def get_data(self, ix):
        img = np.load(self.images[ix])['x'][:-1]
        for band in range(7):
            if self.norm_mode == 'mean_std':
                img[band] = (img[band] - self.stats.loc[band]
                             ['mean']) / self.stats.loc[band]['std']
            elif self.norm_mode == 'min_max':
                img[band] = (img[band] - self.stats.loc[band]['min']) / \
                    (self.stats.loc[band]['max'] - self.stats.loc[band]['min'])
        return img

    def __getitem__(self, ix):
        img = self.get_data(ix)
        label = self.labels[ix]
        # label = int(label)
        if self.trans is not None:
            img = np.transpose(img, (1, 2, 0))
            img = self.trans(image=img)['image']
            img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img),  torch.tensor(label).float()


class AllDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, years=None, path='data', stats_path='real_stats.csv', norm_mode='min_max', trans=None):
        self.images = images
        self.labels = labels
        self.years = years
        self.trans = trans
        self.path = path
        self.norm_mode = norm_mode
        self.stats = pd.read_csv(f'{self.path}/{stats_path}', index_col=0)

    def __len__(self):
        return len(self.images)

    def get_data(self, ix):
        img = np.load(self.images[ix])['x']
        nl_img = np.zeros((9, 255, 255), dtype=img.dtype)
        for band in range(7):
            nl_img[band] = (img[band] - self.stats.loc[band]['min']) / \
                (self.stats.loc[band]['max'] - self.stats.loc[band]['min'])
        # dmsp
        if self.years[ix] < 2012:
            nl_img[7] = (img[7] - self.stats.loc[7]['min']) / \
                (self.stats.loc[7]['max'] - self.stats.loc[7]['min'])
        else:
            nl_img[8] = (img[7] - self.stats.loc[8]['min']) / \
                (self.stats.loc[8]['max'] - self.stats.loc[8]['min'])
        return nl_img

    def __getitem__(self, ix):
        img = self.get_data(ix)
        label = self.labels[ix]
        # label = int(label)
        if self.trans is not None:
            img = np.transpose(img, (1, 2, 0))
            img = self.trans(image=img)['image']
            img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img),  torch.tensor(label).float()


class NonNLDatasetCoordCountry(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, countrys=None, coords=None, path='data', stats_path='real_stats.csv', norm_mode='min_max', trans=None):
        self.images = images
        self.labels = labels
        self.trans = trans
        self.path = path
        self.norm_mode = norm_mode
        self.stats = pd.read_csv(f'{self.path}/{stats_path}', index_col=0)
        self.countrys = countrys
        self.coords = coords

    def __len__(self):
        return len(self.images)

    def get_data(self, ix):
        img = np.load(self.images[ix])['x'][:-1]
        for band in range(7):
            if self.norm_mode == 'mean_std':
                img[band] = (img[band] - self.stats.loc[band]
                             ['mean']) / self.stats.loc[band]['std']
            elif self.norm_mode == 'min_max':
                img[band] = (img[band] - self.stats.loc[band]['min']) / \
                    (self.stats.loc[band]['max'] - self.stats.loc[band]['min'])
        return img

    def __getitem__(self, ix):
        img = self.get_data(ix)
        label = self.labels[ix]
        coord = self.coords[ix]
        country = COUNTRYS.index(self.countrys[ix])
        if self.trans is not None:
            img = np.transpose(img, (1, 2, 0))
            img = self.trans(image=img)['image']
            img = np.transpose(img, (2, 0, 1))

        data = {'img': torch.from_numpy(img), 'coord': torch.from_numpy(
            coord).float(), 'label': torch.tensor(label).float(), 'country': country}

        return data


class AllDatasetCoordCountry(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, years=None, countrys=None, coords=None, path='data', stats_path='real_stats.csv', norm_mode='min_max', trans=None):
        self.images = images
        self.labels = labels
        self.years = years
        self.trans = trans
        self.path = path
        self.norm_mode = norm_mode
        self.stats = pd.read_csv(f'{self.path}/{stats_path}', index_col=0)
        self.countrys = countrys
        self.coords = coords

    def __len__(self):
        return len(self.images)

    def get_data(self, ix):
        img = np.load(self.images[ix])['x']
        nl_img = np.zeros((9, 255, 255), dtype=img.dtype)
        for band in range(7):
            nl_img[band] = (img[band] - self.stats.loc[band]['min']) / \
                (self.stats.loc[band]['max'] - self.stats.loc[band]['min'])
        # dmsp
        if self.years[ix] < 2012:
            nl_img[7] = (img[7] - self.stats.loc[7]['min']) / \
                (self.stats.loc[7]['max'] - self.stats.loc[7]['min'])
        else:
            nl_img[8] = (img[7] - self.stats.loc[8]['min']) / \
                (self.stats.loc[8]['max'] - self.stats.loc[8]['min'])
        return nl_img

    def __getitem__(self, ix):
        img = self.get_data(ix)
        label = self.labels[ix]
        coord = self.coords[ix]
        country = COUNTRYS.index(self.countrys[ix])
        if self.trans is not None:
            img = np.transpose(img, (1, 2, 0))
            img = self.trans(image=img)['image']
            img = np.transpose(img, (2, 0, 1))

        data = {'img': torch.from_numpy(img), 'coord': torch.from_numpy(
            coord).float(), 'label': torch.tensor(label).float(), 'country': country}

        return data


class AllDatasetCoord(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, years=None, coords=None, path='data', stats_path='real_stats.csv', norm_mode='min_max', trans=None):
        self.images = images
        self.labels = labels
        self.years = years
        self.trans = trans
        self.path = path
        self.norm_mode = norm_mode
        self.stats = pd.read_csv(f'{self.path}/{stats_path}', index_col=0)
        self.coords = coords

    def __len__(self):
        return len(self.images)

    def get_data(self, ix):
        img = np.load(self.images[ix])['x']
        nl_img = np.zeros((9, 255, 255), dtype=img.dtype)
        for band in range(7):
            nl_img[band] = (img[band] - self.stats.loc[band]['min']) / \
                (self.stats.loc[band]['max'] - self.stats.loc[band]['min'])
        # dmsp
        if self.years[ix] < 2012:
            nl_img[7] = (img[7] - self.stats.loc[7]['min']) / \
                (self.stats.loc[7]['max'] - self.stats.loc[7]['min'])
        else:
            nl_img[8] = (img[7] - self.stats.loc[8]['min']) / \
                (self.stats.loc[8]['max'] - self.stats.loc[8]['min'])
        return nl_img

    def __getitem__(self, ix):
        img = self.get_data(ix)
        label = self.labels[ix]
        coord = self.coords[ix]
        if self.trans is not None:
            img = np.transpose(img, (1, 2, 0))
            img = self.trans(image=img)['image']
            img = np.transpose(img, (2, 0, 1))

        data = {'img': torch.from_numpy(img), 'coord': torch.from_numpy(
            coord).float(), 'label': torch.tensor(label).float()}

        return data


class NonNLDatasetCoord(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, coords=None, path='data', stats_path='real_stats.csv', norm_mode='min_max', trans=None):
        self.images = images
        self.labels = labels
        self.trans = trans
        self.path = path
        self.norm_mode = norm_mode
        self.stats = pd.read_csv(f'{self.path}/{stats_path}', index_col=0)
        self.coords = coords

    def __len__(self):
        return len(self.images)

    def get_data(self, ix):
        img = np.load(self.images[ix])['x'][:-1]
        for band in range(7):
            if self.norm_mode == 'mean_std':
                img[band] = (img[band] - self.stats.loc[band]
                             ['mean']) / self.stats.loc[band]['std']
            elif self.norm_mode == 'min_max':
                img[band] = (img[band] - self.stats.loc[band]['min']) / \
                    (self.stats.loc[band]['max'] - self.stats.loc[band]['min'])
        return img

    def __getitem__(self, ix):
        img = self.get_data(ix)
        label = self.labels[ix]
        coord = self.coords[ix]
        if self.trans is not None:
            img = np.transpose(img, (1, 2, 0))
            img = self.trans(image=img)['image']
            img = np.transpose(img, (2, 0, 1))

        data = {'img': torch.from_numpy(img), 'coord': torch.from_numpy(
            coord).float(), 'label': torch.tensor(label).float()}

        return data


class RGBDatasetCoord(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, coords=None, path='data', stats_path='real_stats.csv', norm_mode='min_max', trans=None):
        self.images = images
        self.labels = labels
        self.trans = trans
        self.path = path
        self.norm_mode = norm_mode
        self.stats = pd.read_csv(f'{self.path}/{stats_path}', index_col=0)
        self.coords = coords

    def __len__(self):
        return len(self.images)

    def get_data(self, ix):
        img = np.load(self.images[ix])['x'][:3]
        for band in range(3):
            if self.norm_mode == 'mean_std':
                img[band] = (img[band] - self.stats.loc[band]
                             ['mean']) / self.stats.loc[band]['std']
            elif self.norm_mode == 'min_max':
                img[band] = (img[band] - self.stats.loc[band]['min']) / \
                    (self.stats.loc[band]['max'] - self.stats.loc[band]['min'])
        return img

    def __getitem__(self, ix):
        img = self.get_data(ix)
        label = self.labels[ix]
        coord = self.coords[ix]
        if self.trans is not None:
            img = np.transpose(img, (1, 2, 0))
            img = self.trans(image=img)['image']
            img = np.transpose(img, (2, 0, 1))

        data = {'img': torch.from_numpy(img), 'coord': torch.from_numpy(
            coord).float(), 'label': torch.tensor(label).float()}

        return data


class RGBDatasetCoordCountry(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, countrys=None, coords=None, path='data', stats_path='real_stats.csv', norm_mode='min_max', trans=None):
        self.images = images
        self.labels = labels
        self.trans = trans
        self.path = path
        self.norm_mode = norm_mode
        self.stats = pd.read_csv(f'{self.path}/{stats_path}', index_col=0)
        self.countrys = countrys
        self.coords = coords

    def __len__(self):
        return len(self.images)

    def get_data(self, ix):
        img = np.load(self.images[ix])['x'][:3]
        for band in range(3):
            if self.norm_mode == 'mean_std':
                img[band] = (img[band] - self.stats.loc[band]
                             ['mean']) / self.stats.loc[band]['std']
            elif self.norm_mode == 'min_max':
                img[band] = (img[band] - self.stats.loc[band]['min']) / \
                    (self.stats.loc[band]['max'] - self.stats.loc[band]['min'])
        return img

    def __getitem__(self, ix):
        img = self.get_data(ix)
        label = self.labels[ix]
        coord = self.coords[ix]
        country = COUNTRYS.index(self.countrys[ix])
        if self.trans is not None:
            img = np.transpose(img, (1, 2, 0))
            img = self.trans(image=img)['image']
            img = np.transpose(img, (2, 0, 1))

        data = {'img': torch.from_numpy(img), 'coord': torch.from_numpy(
            coord).float(), 'label': torch.tensor(label).float(), 'country': country}

        return data
