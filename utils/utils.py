# -*- coding: utf-8 -*-
import numpy as np
import torch, math
from minepy import MINE
from itertools import combinations
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import KernelDensity
from sklearn.metrics import cohen_kappa_score

def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return m.mic()

def get_pair(x,Nw):
    Quotient = x//(2*Nw)
    Remainder = x%(2*Nw)
    # if Nw==2:
    #     pair = (3 - Remainder) + Quotient * 4
    # if Nw==4:
    #     pair = (7 - Remainder) + Quotient * 8
    pair = (2 * Nw - 1 - Remainder) + Quotient * 2 * Nw
    return pair

def select_consider_pair(x, Ns, Nw):
    select_set = []
    count = 0
    point=0
    while(count<2*Ns):
        if x[point] not in select_set:
            select_set.append(x[point])
            count += 1
        else:
            point += 1
            continue
        pair = get_pair(x[point], Nw)
        if pair not in select_set:
            select_set.append(pair)
            count += 1
        point += 1
    return select_set

def Mutual_Information(X, Y):
    I_y_x_total = []
    for col in range(X.shape[1]):
        f = X[:, col].reshape([-1, 1])
        std = np.std(f, ddof=1)
        h_opt = math.pow(4/(3*f.shape[0]), 1/5)*std

        classes = set(Y)
        H_y = 0
        H_y_x = []

        Prob_x_y = {}  # 似然概率
        Prob_y = {}  # 先验概率
        Prob_x = np.zeros([X.shape[0]])
        Prob_y_x = {}  # 后验概率
        for i in classes:
            p_y = np.mean(Y == i)
            Prob_y[i] = p_y
            if p_y != 0:
                H_y = H_y - p_y * np.log(p_y)

            index = (Y == i)
            kde = KernelDensity(bandwidth=h_opt, kernel='gaussian').fit(f[index])
            p_x_y = np.exp(kde.score_samples(f))
            Prob_x_y[i] = p_x_y
            Prob_x = Prob_x + p_x_y * p_y

        for i in classes:
            Prob_y_x[i] = Prob_x_y[i] * Prob_y[i] / Prob_x
            H_y_x.append(-Prob_y_x[i] * np.log(Prob_y_x[i]))

        H_y_x = np.sum(H_y_x)
        I_y_x = H_y - H_y_x
        I_y_x_total.append(I_y_x)
    return I_y_x_total

class parzen_window_classifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        y = y.astype(int)
        # Store the classes seen during fit
        self.classes = set(y)
        self.X_ = X
        self.y_ = y
        kde_class = {}
        Prob_y = np.zeros([len(self.classes)])  # 先验概率
        for i in self.classes:
            p_y = np.mean(y == i)
            Prob_y[i] = p_y
            index = (y == i)
            kde_cols = {}
            for col in range(X.shape[1]):
                f = X[:, col].reshape([-1, 1])
                std = np.std(f, ddof=1)
                h_opt = math.pow(4 / (3 * f.shape[0]), 1 / 5) * std
                kde_cols[col] = KernelDensity(bandwidth=h_opt, kernel='gaussian').fit(f[index])
            kde_class[i] = kde_cols
        self.Prob_y = Prob_y
        self.kde_class = kde_class
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        Prob_y_x = self.predict_proba(X)
        pred_target = np.argmax(Prob_y_x, axis=1)
        return pred_target

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        return self

    def score(self, X, y, sample_weight=None):
        # score = accuracy_score(Y, self.predict(X))
        kappa = cohen_kappa_score(y, self.predict(X))
        return kappa

    def predict_proba(self, X):
        Prob_x_y = np.zeros([X.shape[0], len(self.classes)])  # 似然概率
        for i in self.classes:
            mul_p = np.ones([X.shape[0]])
            for col in range(X.shape[1]):
                f = X[:, col].reshape([-1, 1])
                p_x_y = np.exp(self.kde_class[i][col].score_samples(f))
                mul_p = mul_p * p_x_y
            Prob_x_y[:, i] = mul_p

        Prob_xy = Prob_x_y * self.Prob_y  # 联合概率
        Prob_x = np.sum(Prob_xy, axis=1).reshape(-1, 1)
        Prob_x = np.tile(Prob_x, reps=(1, len(self.classes)))
        Prob_y_x = Prob_xy / Prob_x # 后验概率
        return Prob_y_x

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + \
                      vectors.pow(2).sum(dim=1).view(1,-1) + \
                      vectors.pow(2).sum(dim=1).view(-1,1)
    return distance_matrix

class TripletSelector:
    '''
    Implementation should return indices of anchors, positive and negative samples
    return torch numpy array of shape [N_triplets x 3]
    '''
    def __init__(self):
        pass
    def get_triplets(self, embeddings, labels):
        raise NotImplementedError

def random_hard_negative(loss_values, margin):
    hard_negatives = np.where(loss_values + margin > 0 )[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None
    # return hard_negatives if len(hard_negatives) > 0 else None

class FunctionNegativeTriplerSelector(TripletSelector):
    def __init__(self, margin, negative_selection_fn, cuda):
        super(FunctionNegativeTriplerSelector, self).__init__()
        self.cuda = cuda
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cuda:
            embeddings = embeddings.cuda()
        distance_matrix = pdist(embeddings)
        triplets = []
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))
            anchor_positives = np.array(anchor_positives)
            ap_distances = distance_matrix[anchor_positives[:,0], anchor_positives[:,1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[anchor_positive[0], negative_indices]
                loss_values = loss_values.cpu().data.numpy()
                hard_negatives = self.negative_selection_fn(loss_values, self.margin)
                if hard_negatives is not None:
                    if not type(hard_negatives) in (tuple, list):
                        hard_negatives = (hard_negatives,)
                    for hard_negative in hard_negatives:
                        hard_negative = negative_indices[hard_negative]
                        triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])
        # if len(triplets) == 0:
        #      triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])
        triplets = np.array(triplets)
        return torch.LongTensor(triplets)

def RandomNegativeTripletSelector(margin, cuda=False):
    return FunctionNegativeTriplerSelector(margin, random_hard_negative, cuda)