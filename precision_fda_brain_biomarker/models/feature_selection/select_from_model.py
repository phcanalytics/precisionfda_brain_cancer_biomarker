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
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from precision_fda_brain_biomarker.models.baselines.base_model import PickleableBaseModel


class SVCSelect(SelectFromModel, PickleableBaseModel):
    def __init__(self, estimator=SVC(), threshold=None, prefit=False, norm_order=1, max_features=None):
        super(SVCSelect, self).__init__(estimator, threshold=threshold, prefit=prefit, norm_order=norm_order,
                                        max_features=max_features)


class LDASelect(SelectFromModel, PickleableBaseModel):
    def __init__(self, estimator=LinearDiscriminantAnalysis(), threshold=None,
                 prefit=False, norm_order=1, max_features=None):
        super(LDASelect, self).__init__(estimator, threshold=threshold, prefit=prefit, norm_order=norm_order,
                                        max_features=max_features)


class LSVCSelect(SelectFromModel, PickleableBaseModel):
    def __init__(self, estimator=LinearSVC(), threshold=None, prefit=False, norm_order=1, max_features=None):
        super(LSVCSelect, self).__init__(estimator, threshold=threshold, prefit=prefit, norm_order=norm_order,
                                         max_features=max_features)


class LassoSelect(SelectFromModel, PickleableBaseModel):
    def __init__(self, estimator=LogisticRegression(penalty='l1', solver='liblinear'), threshold=None, prefit=False,
                 norm_order=1, max_features=None):
        super(LassoSelect, self).__init__(estimator, threshold=threshold, prefit=prefit, norm_order=norm_order,
                                          max_features=max_features)


class RidgeSelect(SelectFromModel, PickleableBaseModel):
    def __init__(self, estimator=LogisticRegression(penalty='l2'), threshold=None, prefit=False, norm_order=1,
                 max_features=None):
        super(RidgeSelect, self).__init__(estimator, threshold=threshold, prefit=prefit, norm_order=norm_order,
                                          max_features=max_features)