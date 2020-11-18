import scipy, os
import scipy.io as scio
import scipy.signal as signal
import numpy as np
from utils.datasets import filter
from utils.utils import select_consider_pair, Mutual_Information

Nw=2
Ns=4
fs = 250
session_select_param_total = {}
l_freq_list = [6, 8, 10, 12, 14, 17, 20, 23, 26, 29, 32, 35]
h_freq_list = [8, 10, 12, 14, 19, 22, 25, 28, 31, 34, 37, 40]

sessions=['A01']
# sessions=['A01','A02','A03','A04','A05','A06','A07','A08','A09']
for session in sessions:
    filename = '../../SCI/raw_data/bci2a_data/' + session + 'T.mat'
    data = scio.loadmat(filename)
    trials_raw = data['trials_regress_eog']
    trials_raw = np.expand_dims(trials_raw, 1)
    trials_transpose = np.transpose(trials_raw,axes=(0,1,3,2)) #[288, 1, 22, 876]
    labels = data['labels'].squeeze()

    session_select_param = {}
    for label in set(labels):
        label_mask = (labels == label)
        positive_indices = np.where(label_mask)[0]
        negative_indices = np.where(np.logical_not(label_mask))[0]
        trials_R1_raw = trials_transpose[positive_indices,:,:,:]
        trials_R2_raw = trials_transpose[negative_indices,:,:,:]
        X_total = []
        W_spacial = []
        for i in range(len(l_freq_list)):
            l_freq = l_freq_list[i]
            h_freq = h_freq_list[i]

            trials_R1 = filter(trials_R1_raw, l_freq=l_freq, h_freq=h_freq, fs=fs, axis=3)
            trials_R2 = filter(trials_R2_raw, l_freq=l_freq, h_freq=h_freq, fs=fs, axis=3)

            cov_R1 = np.zeros([trials_R1.shape[0],trials_R1.shape[2], trials_R1.shape[2]])
            for i in range(trials_R1.shape[0]):
                trials_R1_temp = trials_R1[i,0,:,:].squeeze()
                maxtrix_multiply = np.dot(trials_R1_temp, trials_R1_temp.T)
                trace = np.trace(maxtrix_multiply)
                cov_R1[i,:,:] = maxtrix_multiply/trace
            cov_R1 = np.mean(cov_R1,axis=0,keepdims=False)

            cov_R2 = np.zeros([trials_R2.shape[0], trials_R2.shape[2], trials_R2.shape[2]])
            for i in range(trials_R2.shape[0]):
                trials_R2_temp = trials_R2[i, 0, :, :].squeeze()
                maxtrix_multiply = np.dot(trials_R2_temp, trials_R2_temp.T)
                trace = np.trace(maxtrix_multiply)
                cov_R2[i, :, :] = maxtrix_multiply / trace
            cov_R2 = np.mean(cov_R2, axis=0, keepdims=False)

            D1, W1 = scipy.linalg.eig(cov_R1, cov_R1+cov_R2)
            index = np.argsort(D1)[::-1]
            D1 = D1[index]
            W1 = W1[:,index]
            index = np.concatenate((np.arange(0,Nw),np.arange(-Nw,0)),axis=0)
            W1_select = W1[:,index]
            W_spacial.append(W1_select.T)

            #计算能量特征
            fv_R1 = np.zeros([trials_R1.shape[0], 2*Nw])
            for i in range(trials_R1.shape[0]):
                maxtrix_multiply = np.dot(W1_select.T, trials_R1[i,0,:,:])
                maxtrix_multiply = np.dot(maxtrix_multiply,maxtrix_multiply.T)
                fv_R1[i,:] = np.log(np.diag(maxtrix_multiply)/np.trace((maxtrix_multiply)))

            fv_R2 = np.zeros([trials_R2.shape[0], 2 * Nw])
            for i in range(trials_R2.shape[0]):
                maxtrix_multiply = np.dot(W1_select.T, trials_R2[i, 0, :, :])
                maxtrix_multiply = np.dot(maxtrix_multiply, maxtrix_multiply.T)
                fv_R2[i, :] = np.log(np.diag(maxtrix_multiply) / np.trace((maxtrix_multiply)))
            X = np.concatenate((fv_R1, fv_R2),axis=0)
            X_total.append(X)
        W_spacial = np.concatenate(W_spacial, axis=0) #shape:[9*2*Nw, 22]
        X_total = np.concatenate(X_total, axis=1) #shape:[288, 9*2*Nw]
        Y = np.array([0]*trials_R1_raw.shape[0]+[1]*trials_R2_raw.shape[0])
        # scores = np.array(list(map(lambda x:mic(x, Y), X_total.T)))
        scores = np.array(Mutual_Information(X_total, Y))
        index = np.argsort(scores)[::-1]
        select_set = select_consider_pair(index, Ns, Nw)
        print([(l_freq_list[i // (2 * Nw)], h_freq_list[i // (2 * Nw)]) for i in select_set])
        session_select_param[label] = {'W_spacial':W_spacial, 'select_set':select_set}
    session_select_param_total[session] = session_select_param
    print('session: {0} has computed W_spatical!'.format(session))

for session in sessions:
    session_select_param = session_select_param_total[session]
    '''Train set prepocess and feacture selection'''
    filename = '../../SCI/raw_data/bci2a_data/' + session + 'T.mat'
    data = scio.loadmat(filename)
    trials_raw = data['trials_regress_eog']
    # trials_raw = data['trials_split']
    trials_raw = np.expand_dims(trials_raw, 1)
    trials_transpose = np.transpose(trials_raw,axes=(0,1,3,2)) #[288, 1, 22, 876]
    labels = data['labels'].squeeze()
    trials_filt = []
    trials_select = []
    for i in range(len(l_freq_list)):
        l_freq = l_freq_list[i]
        h_freq = h_freq_list[i]
        trials_bin = filter(trials_transpose, l_freq=l_freq, h_freq=h_freq, fs=fs, axis=3)
        trials_filt.append(trials_bin)
    trials_filt = np.concatenate(trials_filt, 2)
    for i in range(len(session_select_param)):
        W_spacial = session_select_param[i]['W_spacial']
        select_set = session_select_param[i]['select_set']
        trials_Wspacial = np.zeros([trials_filt.shape[0], 1, W_spacial.shape[0], trials_transpose.shape[3]])
        #shape:[288,1,36,876]
        for j in range(trials_filt.shape[0]):
            for k in range(9):
                trials_Wspacial[j,0,2*Nw*k:2*Nw*(k+1),:] = np.dot(W_spacial[2*Nw*k:2*Nw*(k+1),:],
                                                                  trials_filt[j,0,22*k:22*(k+1),:])
        trials_select.append(trials_Wspacial[:, :, select_set,:])
    trials_select = np.concatenate(trials_select,axis=2)
    trials_select_hilbert = signal.hilbert(trials_select, axis=-1)
    trails_select_envelop = np.abs(trials_select_hilbert)
    trials_select_resample = signal.resample(trails_select_envelop, 70, axis=-1) #[288, 1, 32, 70]

    train_filename = './data/filter_dataset/train_dataset/{0}T.mat'.format(session)
    train_dataset = {'trials': trials_select_resample, 'labels': labels}
    if not os.path.exists(os.path.dirname(train_filename)):
        os.makedirs(os.path.dirname(train_filename))
    scio.savemat(train_filename, train_dataset)

    '''Test set prepocess and feacture selection'''
    filename = '../../SCI/raw_data/bci2a_data/' + session + 'E.mat'
    data = scio.loadmat(filename)
    trials_raw = data['trials_regress_eog']
    # trials_raw = data['trials_split']
    trials_raw = np.expand_dims(trials_raw, 1)
    trials_transpose = np.transpose(trials_raw, axes=(0, 1, 3, 2))  # [288, 1, 22, 876]
    labels = data['labels'].squeeze()
    trials_filt = []
    trials_select = []
    for i in range(len(l_freq_list)):
        l_freq = l_freq_list[i]
        h_freq = h_freq_list[i]
        trials_bin = filter(trials_transpose, l_freq=l_freq, h_freq=h_freq, fs=fs, axis=3)
        trials_filt.append(trials_bin)
    trials_filt = np.concatenate(trials_filt, 2)
    for i in range(len(session_select_param)):
        W_spacial = session_select_param[i]['W_spacial']
        select_set = session_select_param[i]['select_set']
        trials_Wspacial = np.zeros([trials_filt.shape[0], 1, W_spacial.shape[0], trials_transpose.shape[3]])
        # shape:[288,1,36,876]
        for j in range(trials_filt.shape[0]):
            for k in range(9):
                trials_Wspacial[j, 0, 2 * Nw * k:2 * Nw * (k + 1), :] = np.dot(
                    W_spacial[2 * Nw * k:2 * Nw * (k + 1), :],
                    trials_filt[j, 0, 22 * k:22 * (k + 1), :])
        trials_select.append(trials_Wspacial[:, :, select_set, :])
    trials_select = np.concatenate(trials_select, axis=2)
    trials_select_hilbert = signal.hilbert(trials_select, axis=-1)
    trails_select_envelop = np.abs(trials_select_hilbert)
    trials_select_resample = signal.resample(trails_select_envelop, 70, axis=-1) #[288, 1, 32, 70]

    test_filename = './data/filter_dataset/test_dataset/{0}E.mat'.format(session)
    test_dataset = {'trials': trials_select_resample, 'labels': labels}
    if not os.path.exists(os.path.dirname(test_filename)):
        os.makedirs(os.path.dirname(test_filename))
    scio.savemat(test_filename, test_dataset)

    print('session: {0} has Finished!'.format(session))

print('All Finish!')