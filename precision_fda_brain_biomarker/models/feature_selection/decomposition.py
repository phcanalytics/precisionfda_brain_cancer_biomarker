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
from sklearn import decomposition
from precision_fda_brain_biomarker.models.baselines.base_model import PickleableBaseModel


class DecompositionComponents(object):
    @staticmethod
    def get_hyperparameter_ranges():
        ranges = {
            "pca_n_components": (10, 20, 50, 100)
        }
        return ranges


class FastICA(decomposition.FastICA, PickleableBaseModel, DecompositionComponents):
    pass


class KPCA(decomposition.KernelPCA, PickleableBaseModel, DecompositionComponents):
    pass


class IPCA(decomposition.IncrementalPCA, PickleableBaseModel, DecompositionComponents):
    pass


class NMF(decomposition.NMF, PickleableBaseModel, DecompositionComponents):
    pass


class SparseCoder(decomposition.SparseCoder, PickleableBaseModel, DecompositionComponents):
    pass


class FactorAnalysis(decomposition.FactorAnalysis, PickleableBaseModel, DecompositionComponents):
    pass


class TruncatedSVD(decomposition.TruncatedSVD, PickleableBaseModel, DecompositionComponents):
    pass


class Dirichlet(decomposition.LatentDirichletAllocation, PickleableBaseModel, DecompositionComponents):
    pass


class Dictionary(decomposition.DictionaryLearning, PickleableBaseModel, DecompositionComponents):
    pass

