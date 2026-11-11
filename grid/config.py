# --- 1. Cross-validation ---
# Outer split of repeat r: random_state = seed + r. Task (r, f): seed + n_folds*r + f.
# Inner fold k of that task: (task seed + f) + INNER_SEED_STRIDE * k.
CV = {
    "seed": 0,
    "n_repeats": 5,
    "n_outer_folds": 5,
    "n_inner_folds": 5,
}
INNER_SEED_STRIDE = 100

# --- 2. Arms ---
# 13 arms: Original + six generators without GRID + the same six with GRID ("grid": True).
# Every generator augments each class up to the majority-class count (ADASYN approximately).
EXPERIMENTS = [
    {"name": "1. Baseline", "algo": "base"},

    {"name": "2-1. SMOTE", "algo": "smote", "grid": False},
    {"name": "2-2. ADASYN", "algo": "adasyn", "grid": False},
    {"name": "2-3. ROS (Raw)", "algo": "ros", "grid": False},
    {"name": "3-1. CTGAN (Raw)", "algo": "ctgan", "grid": False},
    {"name": "3-2. TVAE (Raw)", "algo": "tvae", "grid": False},
    {"name": "3-3. COPULAGAN (Raw)", "algo": "copulagan", "grid": False},

    {"name": "4-1. SMOTE-GRID", "algo": "smote", "grid": True},
    {"name": "4-2. ADASYN-GRID", "algo": "adasyn", "grid": True},
    {"name": "4-3. ROS-GRID", "algo": "ros", "grid": True},
    {"name": "5-1. CTGAN-GRID", "algo": "ctgan", "grid": True},
    {"name": "5-2. TVAE-GRID", "algo": "tvae", "grid": True},
    {"name": "5-3. COPULAGAN-GRID", "algo": "copulagan", "grid": True},
]

# --- 3. Base generators ---
# SMOTE / SMOTENC / ADASYN: imbalanced-learn defaults (k_neighbors / n_neighbors = 5).
# CTGAN / TVAE / CopulaGAN: used when the --metadata YAML has no `generators` entry.
DEEP_GENERATOR_DEFAULTS = {
    "ctgan": {"epochs": 300, "batch_size": 500, "pac": 10, "generator_dim": (256, 256),
              "discriminator_dim": (256, 256), "generator_lr": 2e-4, "discriminator_lr": 2e-4,
              "generator_decay": 1e-6, "discriminator_decay": 1e-6},
    "tvae": {"epochs": 300, "batch_size": 500, "embedding_dim": 128, "compress_dims": (128, 128),
             "decompress_dims": (128, 128), "loss_factor": 2.0},
    "copulagan": {"epochs": 300, "batch_size": 500, "generator_dim": (256, 256), "discriminator_dim": (256, 256),
                  "generator_lr": 2e-4, "discriminator_lr": 2e-4, "generator_decay": 1e-6,
                  "discriminator_decay": 1e-6},
}
DEEP_SAMPLING = {
    "batch_size": 5000,
    "max_tries_per_batch": 200,
}

# --- 4. GRID refinement ---
# Phase I teacher: TabPFN v2 (Prior-Labs/TabPFN-v2-clf), fitted once per inner fold.
TEACHER_CHECKPOINT = "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
TEACHER_PARAMS = {"n_estimators": 16}

GRID_PARAMS = {
    # Phase I: student distilled from the teacher on the training set + noisy copies
    "n_distill_noise": 5,
    "distill_noise_std": 0.03,  # noise SD (continuous predictors only)
    "student_hidden": 256,
    "student_dropout": 0.1,
    "distill_lr": 1e-3,
    "distill_epochs": 200,
    "distill_patience": 20,
    "distill_val_frac": 0.2,  # validation share for early stopping
    # Phase II: one autoencoder per class on the real samples of that class
    "ae_hidden": 256,
    "ae_latent": 16,
    "ae_lr": 0.01,
    "ae_epochs": 500,
    "ae_patience": 20,
    "ae_min_class_size": 5,
    # Phase III: K Adam steps on every synthetic sample
    "steps": 30,  # K
    "lr": 0.01,
    "alpha_cls": 1.0,  # weight of the student loss
    "beta_rec": 1.0,  # weight of the autoencoder loss, rescaled at every step by the gradient-norm ratio
    "cat_init_logit": 0.5,  # initial logit of the generated category (others 0)
    "tau_start": 1.0,  # Gumbel-softmax temperature, annealed linearly
    "tau_end": 0.1,
}

# --- 5. Downstream classifiers ---
# PyTabKit models (XGB, RF, RealMLP, FTT, TabM): tuned defaults (TD / D).
CLF_NAMES = ["LogisticReg", "TabPFN", "XGB", "RF", "RealMLP", "FTT", "TabM"]
# TabPFN classifier: TabPFN-2.5 (Prior-Labs/tabpfn_2_5)
TABPFN_CLASSIFIER_CHECKPOINT = "tabpfn-v2.5-classifier-v2.5_default.ckpt"
CLASSIFIER_PARAMS = {
    "LogisticReg": {"class_weight": "balanced", "max_iter": 500},
    "TabPFN": {"n_estimators": 16},
}
