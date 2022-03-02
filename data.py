import ast, json
import pandas as pd
import numpy as np
from PIL import Image
import tables
from torch.utils import data
from torchvision import transforms
from collections import Counter

def split_dataset(file_path, label_list=None):
    df = pd.read_csv(file_path, sep='\t')
    df = df.drop_duplicates(subset='current_image_id', keep="last")
    # Keep specific labels
    if label_list: df = df[df['comparison'].isin(label_list)]
    train_split = pd.read_csv('/home/ilourentzou/ChestXGenome/splits/train.csv')
    valid_split = pd.read_csv('/home/ilourentzou/ChestXGenome/splits/valid.csv')
    test_split = pd.read_csv('/home/ilourentzou/ChestXGenome/splits/test.csv')
    pid = list(train_split['dicom_id'].unique())
    train = df[df['current_image_id'].isin(pid)]
    #train2 = df[df['previous_image_id'].isin(pid)] #either works
    pid = list(valid_split['dicom_id'].unique())
    dev = df[df['current_image_id'].isin(pid)]
    pid = list(test_split['dicom_id'].unique())
    test = df[df['current_image_id'].isin(pid)]
    print(Counter(train['comparison']))
    print(Counter(dev['comparison']))
    print(Counter(test['comparison']))
    return train, dev, test


class H5Reader(object):

    def __init__(self, h5_path, filename_idx=None):
        """For fast reading from a h5 file. A file name to index dict is created for fast indexing.
        :param h5_path: the h5 file path to read from.
        """
        self.h5_path = h5_path
        self.h5_file = tables.open_file(self.h5_path, 'r')
        self.data = self.h5_file.root.data
        self.header = self.h5_file.root.header if '/header' in self.h5_file else None
        if filename_idx is None:
            self.filename_idx = dict(enumerate(self.h5_file.root.filename.iterrows()))
            self.filename_idx = dict(zip(self.filename_idx.values(), self.filename_idx.keys()))
            #{name: i for i, name in enumerate(self.h5_file.root.filename.iterrows())}
        else:
            self.filename_idx = filename_idx
        self.h5_file.close()

    def read_image(self, filename, gray_scale=True):
        """Read an image given the file name.
        :param filename: the file name.
        :param gray_scale: if True, convert to gray-scale image.
        :return: the image. None if the file name does not exist.
        """
        self.h5_file = tables.open_file(self.h5_path, 'r')
        self.data = self.h5_file.root.data
        if filename not in self.filename_idx:
            return None
        image = self.data[self.filename_idx[filename]]
        image = image.squeeze()
        if gray_scale and image.ndim == 3:
            image = image.mean(axis=-1).astype(image.dtype)
        self.h5_file.close()
        return image

    def read_header(self, filename):
        """Read the header given the file name.
        :param filename: the file name.
        :return: the with original image shape and resizing info. None if the file name does not exist.
        """
        if filename not in self.filename_idx:
            return None
        header = self.header[self.filename_idx[filename]]
        header = json.loads(header)
        return header.copy()


class ComparisonsDataset(data.Dataset):
    def __init__(self, csv_file, h5file, labelset, transform=None):
        """
        Args:
            csv_file (pd.dataframe): dataframe table containing image names, disease severity category label, and other metadata
            image_dir (string): directory containing all of the image files
            transform (callable, optional): optional transform to be applied on a sample
        """
        self.h5file = H5Reader(h5file)
        self.csv_file = csv_file
        self.transform = transform
        self.labelset = labelset
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        item = self.csv_file.iloc[idx]
        current_image_id = item['current_image_id']
        previous_image_id = item['previous_image_id']
        current_image = self.h5file.read_image(current_image_id+'.dcm', gray_scale=True)
        previous_image = self.h5file.read_image(previous_image_id+'.dcm', gray_scale=True)
        bbox_current = ast.literal_eval(item['bbox_coord_224_subject'])
        bbox_previous = ast.literal_eval(item['bbox_coord_224_object'])
        assert len(bbox_current) == len(bbox_previous) == 4
        current_image = Image.fromarray(current_image/np.max(current_image))
        previous_image = Image.fromarray(previous_image/np.max(previous_image))
        cropped_current = current_image.crop(bbox_current)
        cropped_previous = previous_image.crop(bbox_previous)
        if self.transform is not None:
            cropped_current = self.transform(cropped_current)
            cropped_previous = self.transform(cropped_previous)
        label = item['comparison']
        labelidx = self.labelset.index(label)
        meta = dict(item)
        return cropped_current, cropped_previous, labelidx, meta
        #[batch_size x channels x height x width]


