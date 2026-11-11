from tabpfn import TabPFNClassifier
from tabpfn.model_loading import get_cache_dir

from grid.config import TEACHER_CHECKPOINT, TEACHER_PARAMS


def make_teacher(X_train, y_train, seed, cat_idx, device):
    teacher = TabPFNClassifier(
        model_path=str(get_cache_dir() / TEACHER_CHECKPOINT),
        device=device,
        random_state=seed,
        **TEACHER_PARAMS,
        categorical_features_indices=cat_idx if cat_idx else None,
    )
    teacher.fit(X_train.values, y_train.values)
    return teacher
