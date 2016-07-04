from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import cross_val_score, KFold, ShuffleSplit
import scipy.io as scio
import numpy as np

from load_curet_db import load_image_patches_curet


def from_features_to_classif_scores(labels,features):
    #stack them for learning
    features = features.reshape((len(features),-1))
    # apply pipeline
    n = len(features)
    pipeline = make_pipeline(Normalizer(),StandardScaler(),LogisticRegression(C=1.0))
    cv = ShuffleSplit(n,n_iter=3,test_size=1, train_size=1)
    
    scores = cross_val_score(pipeline,features,labels,cv=5,n_jobs=5)
    print('score:',scores)
    return scores


def load_data(data_file):

    f = scio.loadmat(data_file)
    labels = np.ndarray(f['labels'])
    features = np.ndarray(f['scatterings'])

    return labels,features


def load_and_train_curet(data_path='./curet.mat', loadfeatures=True):

    if loadfeatures:
        labels, scatterings = load_data(data_path)
    else:
        labels, scatterings = load_image_patches_curet(data_path)

    scores = from_features_to_classif_scores(labels,scatterings)

    return scores
