"""
Copyright (C) 2019  F.Hoffmann-La Roche Ltd

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import os
import json
import numpy as np
from sklearn.decomposition import PCA as PCAParent
from sklearn.base import BaseEstimator, TransformerMixin
from precision_fda_brain_biomarker.models.baselines.base_model import PickleableBaseModel


class TierConfig(object):
    config_file = os.path.join(os.path.dirname(__file__), 'tier_config.json')
    tier_lists = json.load(open(config_file))

    def __init__(self):
        self.kept_indices = []

    def get_indices(self, tier, feature_names):
        kept = []
        for t in tier.split(','):
            kept.extend(self.tier_lists[t])
        # Note that it is important to sort the indices here, so that they match up with their locations
        # in feature_names in subsequent pipeline steps, if still present and previous features have not been touched.
        return list(sorted([feature_names.index(x) for x in set(kept) if x in feature_names]))


class TierSelect(TransformerMixin, BaseEstimator, PickleableBaseModel, TierConfig):
    def __init__(self, tier="GE", feature_names=[]):
        self.tier = tier
        self.feature_names = feature_names
        self.fitted = False
        super(TierSelect, self).__init__()

    def fit(self, X, y=None):
        self.kept_indices = self.get_indices(self.tier, self.feature_names)
        self.fitted = True
        return self

    def transform(self, X):
        X_transformed = X[:, self.kept_indices]
        return X_transformed


class PartialPCA(PCAParent, PickleableBaseModel, TierConfig):
    def __init__(self, feature_names=[], tier="GE", copy=True,
                 iterated_power='auto', n_components=None, random_state=None,
                 svd_solver='auto', tol=0.0, whiten=False):
        super(PartialPCA, self).__init__(copy=copy, iterated_power=iterated_power, n_components=n_components,
                                         random_state=random_state, svd_solver=svd_solver, tol=tol, whiten=whiten)
        self.feature_names = feature_names
        self.tier = tier
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

    def get_params(self, deep=True):
        ret = super(PartialPCA, self).get_params(deep=deep)
        ret.update({'feature_names': self.feature_names, 'tier': self.tier})
        return ret

    def fit(self, X, y=None):
        self.kept_indices = self.get_indices(self.tier, self.feature_names)
        self.fitted = True
        X_kept = X[:, self.kept_indices]
        return super(PartialPCA, self).fit(X_kept)

    def transform(self, X):
        X_kept = X[:, self.kept_indices]
        ret = super(PartialPCA, self).transform(X_kept)
        X_kept = np.delete(X, self.kept_indices, axis=1)
        X_transformed = np.concatenate((X_kept, ret), axis=1)
        return X_transformed
