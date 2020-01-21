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
from sklearn.pipeline import Pipeline as PipelineParent
from precision_fda_brain_biomarker.models.baselines.base_model import PickleableBaseModel


class Pipeline(PipelineParent, PickleableBaseModel):
    @staticmethod
    def get_feature_selection_stages(stage_names, selection_params={}):
        from precision_fda_brain_biomarker.apps.main import MainApplication

        feature_selection_stages = [MainApplication.get_feature_selection_type_for_name(fs)
                                    for fs in stage_names]

        instances = [cl() for cl in feature_selection_stages]
        available_model_params = [{k: selection_params[k] if k in selection_params else x.get_params()[k]
                                   for k in x.get_params().keys()} for x in instances]

        if "tier" in selection_params:
            tiers_split = selection_params["tier"].split(":")
            instances = [x.set_params(**param) for x, param in zip(instances, available_model_params)]
            for ins, ti in zip(instances, tiers_split):
                if hasattr(ins, "tier"):
                    ins.set_params(tier=ti)

        return instances
