import os
import uuid

from pytabkit.models.nn_models.rtdl_resnet import UniquePrefixCheckpoint
from pytabkit.models.sklearn.sklearn_interfaces import (
    FTT_D_Classifier as FTT,
    RF_SKL_D_Classifier as RF,
    RealMLP_TD_Classifier as RealMLP,
    TabM_D_Classifier as TabM,
    XGB_TD_Classifier as XGB,
)
from sklearn.linear_model import LogisticRegression
from tabpfn import TabPFNClassifier
from tabpfn.model_loading import get_cache_dir

from grid.config import CLASSIFIER_PARAMS, TABPFN_CLASSIFIER_CHECKPOINT


# pytabkit (FTT) names its checkpoint files by id(self), which can collide between parallel workers in
# ./rtdl_checkpoints/. Add PID + uuid to the prefix (only the file name changes, not the training).
def _process_unique_initialize(self):
    self.fn_prefix = f"{os.getpid()}_{uuid.uuid4().hex}_"
    return super(UniquePrefixCheckpoint, self).initialize()


UniquePrefixCheckpoint.initialize = _process_unique_initialize


def build(clf_name, seed, device):
    if clf_name == "TabPFN":
        return TabPFNClassifier(model_path=str(get_cache_dir() / TABPFN_CLASSIFIER_CHECKPOINT), device=device,
                                random_state=seed, **CLASSIFIER_PARAMS["TabPFN"])
    if clf_name == "LogisticReg":
        return LogisticRegression(random_state=seed, n_jobs=1, **CLASSIFIER_PARAMS["LogisticReg"])
    td_params_nn = {"device": device, "random_state": seed, "verbosity": 0}
    td_params_gbdt = {"device": "cpu", "random_state": seed, "verbosity": 0}
    if clf_name == "XGB":
        return XGB(**td_params_gbdt)
    if clf_name == "RF":
        return RF(**td_params_gbdt)
    if clf_name == "RealMLP":
        return RealMLP(**td_params_nn)
    if clf_name == "FTT":
        return FTT(**td_params_nn)
    if clf_name == "TabM":
        return TabM(**td_params_nn)
    raise ValueError(f"unknown classifier: {clf_name}")
