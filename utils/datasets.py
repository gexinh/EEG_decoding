# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split
from scipy.signal import butter, buttord, cheby2, filtfilt, cheb2ord
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

# def filter(trials, l_freq=4.0, h_freq=40.0, fs=250, order=3, axis=2):
#     b,a = butter_bandpass(lowcut=l_freq, highcut=h_freq, fs=fs, order=order)
#     trials_filt = filtfilt(b,a,trials, axis=axis)
#     return trials_filt

# def butter_bandpass(lowcut, highcut, fs, order=3):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(N = order, Wn=[low, high], btype='bandpass')
#     return b, a

def filter(trials, l_freq=8.0, h_freq=30.0, fs=512, axis=3):
    # b, a, order = butter_bandpass(lowcut=l_freq, highcut=h_freq, fs=fs)
    b, a, order = cheby2_bandpass(lowcut=l_freq, highcut=h_freq, fs=fs)
    trials_filt = filtfilt(b,a,trials, axis=axis)
    return trials_filt

def butter_bandpass(lowcut, highcut, fs, gpass=3, gstop=30):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    ws_1 = max(low - 0.1, 1e-2)
    ws_2 = min(high + 0.1, 0.99)
    order, wn = buttord(wp=[low, high], ws=[ws_1, ws_2], gpass=gpass, gstop=gstop)
    b, a = butter(N=order, Wn=wn, btype='bandpass')
    return b, a, order

def cheby2_bandpass(lowcut, highcut, fs, gpass=1, gstop=40):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    ws_1 = max(low - 0.1, 1e-2)
    ws_2 = min(high + 0.1, 0.99)
    order, wn = cheb2ord(wp=[low, high], ws=[ws_1, ws_2], gpass=gpass, gstop=gstop)
    b, a = cheby2(N=order, rs=gstop, Wn=wn, btype='bandpass')
    return b, a, order

def downsample(trials, axis=3, factor=2):
    length = trials.shape[axis]
    index = [i*factor for i in range(length//factor)]
    trials_downsampe = trials[:,:,:,index]
    return trials_downsampe

class TrainDataset(Dataset):
    '''Call by the TripletDataset when Train=True'''
    def __init__(self, session):
        filename = './data/filter_dataset/train_dataset/{0}T.mat'.format(session)
        data = scio.loadmat(filename)
        trials =data['trials']
        labels = data['labels'].squeeze()

        index, _ = train_test_split(np.arange(72), test_size=0.3, random_state=11)
        cla0_index = np.where(labels==0)[0]
        cla1_index = np.where(labels==1)[0]
        cla2_index = np.where(labels == 2)[0]
        cla3_index = np.where(labels == 3)[0]
        train_cla0_index = cla0_index[index]
        train_cla1_index = cla1_index[index]
        train_cla2_index = cla2_index[index]
        train_cla3_index = cla3_index[index]
        train_index = np.concatenate((train_cla0_index,train_cla1_index,train_cla2_index,train_cla3_index),axis=0)

        labels_onehot = np.eye(4)[labels]
        self.trials = torch.tensor(trials[train_index],dtype=torch.float)
        self.labels = torch.tensor(labels_onehot[train_index],dtype=torch.float)

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, item):
        data = (self.trials[item], self.labels[item])
        return data


class ValidDataset(Dataset):
    '''Call by the TripletDataset when Train=Fasle'''

    def __init__(self, session):
        filename = './data/filter_dataset/train_dataset/{0}T.mat'.format(session)
        data = scio.loadmat(filename)
        trials = data['trials']
        labels = data['labels'].squeeze()

        _, index = train_test_split(np.arange(72), test_size=0.3, random_state=11)
        cla0_index = np.where(labels == 0)[0]
        cla1_index = np.where(labels == 1)[0]
        cla2_index = np.where(labels == 2)[0]
        cla3_index = np.where(labels == 3)[0]
        valid_cla0_index = cla0_index[index]
        valid_cla1_index = cla1_index[index]
        valid_cla2_index = cla2_index[index]
        valid_cla3_index = cla3_index[index]
        valid_index = np.concatenate((valid_cla0_index, valid_cla1_index,valid_cla2_index,valid_cla3_index), axis=0)

        labels_onehot = np.eye(4)[labels]
        self.trials = torch.tensor(trials[valid_index], dtype=torch.float)
        self.labels = torch.tensor(labels_onehot[valid_index], dtype=torch.float)

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, item):
        data = (self.trials[item], self.labels[item])
        return data

class TestDataset(Dataset):
    '''Call by the TripletDataset when Train=False'''
    def __init__(self, session):
        filename = './data/filter_dataset/test_dataset/{0}E.mat'.format(session)
        data = scio.loadmat(filename)
        trials = data['trials']
        labels = data['labels'].squeeze()
        labels_onehot = np.eye(4)[labels]
        self.trials = torch.tensor(trials, dtype=torch.float)
        self.labels = torch.tensor(labels_onehot, dtype=torch.int64)

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, item):
        data = (self.trials[item], self.labels[item])
        return data

