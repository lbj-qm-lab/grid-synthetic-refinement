import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split

from grid.data.loading import snap_categoricals


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, input_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x))


class DistilledSurrogate(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1)
                )
    def forward(self, x):
        return self.net(x)


def _make_distill_pool_masked(X_scaled, cat_idx, num_idx, n_noise, noise_std, seed, cont_lo, cont_hi):
    rng = np.random.default_rng(seed)
    pool = [X_scaled]
    for _ in range(n_noise):
        eps = rng.normal(0, noise_std, size=X_scaled.shape)
        eps[:, cat_idx] = 0.0
        Xn = X_scaled + eps
        Xn[:, num_idx] = np.clip(Xn[:, num_idx], cont_lo, cont_hi)
        pool.append(Xn)
    return np.vstack(pool)


class ConditionalRefiner:

    def __init__(self, device, *, steps, lr, alpha_cls, beta_rec, cat_init_logit, tau_start, tau_end,
                 n_distill_noise, distill_noise_std, student_hidden, student_dropout,
                 distill_lr, distill_epochs, distill_patience, distill_val_frac,
                 ae_hidden, ae_latent, ae_lr, ae_epochs, ae_patience, ae_min_class_size):
        self.device = device
        self.lr = lr
        self.steps = steps
        self.alpha = alpha_cls
        self.beta = beta_rec
        self.cat_init_logit = cat_init_logit
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.n_distill_noise = n_distill_noise
        self.distill_noise_std = distill_noise_std
        self.student_hidden = student_hidden
        self.student_dropout = student_dropout
        self.distill_lr = distill_lr
        self.distill_epochs = distill_epochs
        self.distill_patience = distill_patience
        self.distill_val_frac = distill_val_frac
        self.ae_hidden = ae_hidden
        self.ae_latent = ae_latent
        self.ae_lr = ae_lr
        self.ae_epochs = ae_epochs
        self.ae_patience = ae_patience
        self.ae_min_class_size = ae_min_class_size
        self.classifier = None
        self._aes = {}  # class label -> autoencoder

    def _to_clf_onehot(self, X_scaled_arr, X_orig_df):
        parts = [X_scaled_arr[:, self._num_idx]]
        for k, ci in enumerate(self.cat_idx):
            col = np.round(X_orig_df.iloc[:, ci].values.astype(float)).astype(int)
            allowed = np.asarray(self._cat_allowed[k])
            parts.append((col[:, None] == allowed[None, :]).astype(np.float32))  # [N, card]
        return np.concatenate(parts, axis=1).astype(np.float32)

    def fit(self, X_df, y, teacher, *, seed, cat_idx, cat_cols, cat_allowed, teacher_inverse):
        X_scaled = np.asarray(X_df, dtype=np.float64)
        dim = X_scaled.shape[1]

        self.cat_idx = list(cat_idx)
        self._num_idx = [i for i in range(dim) if i not in self.cat_idx]
        # observed range of the continuous predictors (clamped to during refinement)
        self._cont_lo = X_scaled[:, self._num_idx].min(axis=0)
        self._cont_hi = X_scaled[:, self._num_idx].max(axis=0)
        self._cat_allowed_map = cat_allowed
        self._cat_allowed = [np.asarray(cat_allowed[cat_cols[k]]) for k in range(len(self.cat_idx))]

        # Phase I: distil the teacher into the student
        X_pool_scaled = _make_distill_pool_masked(
            X_scaled, self.cat_idx, self._num_idx,
            n_noise=self.n_distill_noise, noise_std=self.distill_noise_std, seed=seed,
            cont_lo=self._cont_lo, cont_hi=self._cont_hi,
        )
        X_pool = X_pool_scaled if teacher_inverse is None else teacher_inverse(X_pool_scaled)
        X_pool_df = pd.DataFrame(X_pool, columns=X_df.columns)
        p_teacher = teacher.predict_proba(X_pool_df)[:, 1].astype(np.float32)

        X_pool_oh = self._to_clf_onehot(X_pool_scaled, X_pool_df)
        dim_oh = X_pool_oh.shape[1]
        self.classifier = DistilledSurrogate(dim_oh, self.student_hidden, self.student_dropout).to(self.device)
        opt = optim.Adam(self.classifier.parameters(), lr=self.distill_lr)

        idx_tr, idx_val = train_test_split(
            np.arange(len(X_pool_oh)), test_size=self.distill_val_frac, random_state=seed
        )
        X_tr_t = torch.tensor(X_pool_oh[idx_tr], dtype=torch.float32, device=self.device)
        p_tr_t = torch.tensor(p_teacher[idx_tr].reshape(-1, 1), dtype=torch.float32, device=self.device)
        X_val_t = torch.tensor(X_pool_oh[idx_val], dtype=torch.float32, device=self.device)
        p_val_t = torch.tensor(p_teacher[idx_val].reshape(-1, 1), dtype=torch.float32, device=self.device)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(self.distill_epochs):
            opt.zero_grad()
            logits = self.classifier(X_tr_t)
            loss = nn.BCEWithLogitsLoss()(logits, p_tr_t)
            loss.backward()
            opt.step()

            with torch.no_grad():
                val_logits = self.classifier(X_val_t)
                val_loss = nn.BCEWithLogitsLoss()(val_logits, p_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.classifier.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.distill_patience:
                    break

        if best_state is None:
            raise RuntimeError("student distillation: the validation loss was never finite")
        self.classifier.load_state_dict(best_state)

        # Phase II: one autoencoder per class
        Xt = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        yt = y.values
        for cls in np.unique(yt):
            Xc = Xt[yt == cls]
            if len(Xc) < self.ae_min_class_size:
                raise ValueError(f"class {cls} has {len(Xc)} real training samples, fewer than "
                                 f"ae_min_class_size={self.ae_min_class_size}: no autoencoder can be fitted")

            ae = AutoEncoder(dim, latent_dim=self.ae_latent, hidden_dim=self.ae_hidden).to(self.device)
            opt = torch.optim.Adam(ae.parameters(), lr=self.ae_lr)
            best = float('inf')
            best_state = None
            pc = 0
            for ep in range(self.ae_epochs):
                opt.zero_grad()
                loss = ((ae(Xc) - Xc) ** 2).mean()
                loss.backward()
                opt.step()
                if loss.item() < best - 1e-6:
                    best = loss.item()
                    best_state = {k: v.clone() for k, v in ae.state_dict().items()}
                    pc = 0
                else:
                    pc += 1
                    if pc >= self.ae_patience:
                        break

            if best_state is None:
                raise RuntimeError(f"autoencoder of class {cls}: the training loss was never finite")
            ae.load_state_dict(best_state)
            ae.eval()
            for p in ae.parameters():
                p.requires_grad_(False)
            self._aes[float(cls)] = ae

        torch.cuda.empty_cache()

    def optimize(self, X_gen_df, y_gen):
        if self.classifier is None:
            raise RuntimeError("ConditionalRefiner.optimize() called before fit()")

        cat_idx = self.cat_idx
        cat_allowed = self._cat_allowed_map
        all_col_names = X_gen_df.columns.tolist()
        num_idx = self._num_idx
        N = len(X_gen_df)
        D = len(all_col_names)

        X_input = np.asarray(X_gen_df, dtype=np.float64)
        X_num = torch.tensor(X_input[:, num_idx], dtype=torch.float32,
                             device=self.device, requires_grad=True)

        # categorical predictors: one logit vector per sample, initialised at the generated category
        cat_logits_list = []
        cat_meta = []  # (col_pos, allowed_values, allowed_values_tensor)

        for col_pos in cat_idx:
            col_name = all_col_names[col_pos]
            allowed = cat_allowed[col_name]
            n_classes = len(allowed)
            allowed_t = torch.tensor(np.asarray(allowed, dtype=float), dtype=torch.float32,
                                     device=self.device)

            current_vals = X_gen_df.iloc[:, col_pos].values
            init_logits = torch.zeros(N, n_classes, device=self.device)
            for row_i, val in enumerate(current_vals):
                match = np.where(allowed == val)[0]
                if len(match) == 0:
                    raise ValueError(f"synthetic `{col_name}` = {val} is not an allowed code {allowed.tolist()}")
                init_logits[row_i, match[0]] = self.cat_init_logit

            logits_param = init_logits.clone().requires_grad_(True)
            cat_logits_list.append(logits_param)
            cat_meta.append((col_pos, allowed, allowed_t))

        optimizer = torch.optim.Adam([X_num] + cat_logits_list, lr=self.lr, weight_decay=0)

        bce = nn.BCEWithLogitsLoss()

        y_t = torch.tensor(y_gen.values, dtype=torch.float32, device=self.device)
        y_target = y_t.unsqueeze(1)

        tau_start, tau_end = self.tau_start, self.tau_end

        def assemble_X(x_num, c_logits_list, tau):
            X_full = torch.zeros(N, D, device=self.device)
            for new_i, orig_i in enumerate(num_idx):
                X_full[:, orig_i] = x_num[:, new_i]
            for logits, (col_pos, _, allowed_t) in zip(c_logits_list, cat_meta):
                probs = F.gumbel_softmax(logits, tau=tau, hard=False)
                X_full[:, col_pos] = (probs * allowed_t.unsqueeze(0)).sum(dim=1)
            return X_full

        def assemble_X_clf(x_num, c_logits_list):
            parts = [x_num]
            for logits in c_logits_list:
                parts.append(F.softmax(logits, dim=1))
            return torch.cat(parts, dim=1)

        alpha, beta = self.alpha, self.beta

        def compute_loss(x_num, c_logits, temperature):
            X_full = assemble_X(x_num, c_logits, temperature)

            logits_out = self.classifier(assemble_X_clf(x_num, c_logits))
            loss_cls = bce(logits_out, y_target)

            loss_rec = 0.0
            n_terms = 0
            for cls, ae in self._aes.items():
                mask = (y_t == float(cls))
                if mask.sum() == 0:
                    continue
                loss_rec = loss_rec + ((ae(X_full[mask]) - X_full[mask]) ** 2).mean()
                n_terms += 1
            loss_rec = loss_rec / n_terms
            return loss_cls, loss_rec

        for i in range(self.steps):
            optimizer.zero_grad()

            tau = tau_start - (tau_start - tau_end) * (i / max(self.steps - 1, 1))
            loss_cls_val, loss_rec_raw = compute_loss(X_num, cat_logits_list, tau)

            if len(cat_logits_list) > 0:
                # separate adaptive weights for the continuous (lam_n) and categorical (lam_c) blocks
                g_cls_n = torch.autograd.grad(loss_cls_val, X_num, retain_graph=True)[0]
                g_rec_n = torch.autograd.grad(loss_rec_raw, X_num, retain_graph=True)[0]
                lam_n = (g_cls_n.norm() / (g_rec_n.norm() + 1e-8)).clamp(max=1e4)

                g_cls_c = torch.autograd.grad(loss_cls_val, cat_logits_list, retain_graph=True)
                g_rec_c = torch.autograd.grad(loss_rec_raw, cat_logits_list, retain_graph=True)
                nc = torch.sqrt(sum(g.pow(2).sum() for g in g_cls_c))
                dc = torch.sqrt(sum(g.pow(2).sum() for g in g_rec_c)) + 1e-8
                lam_c = (nc / dc).clamp(max=1e4)

                X_num.grad = alpha * g_cls_n + beta * lam_n * g_rec_n
                for lg, gc, gr in zip(cat_logits_list, g_cls_c, g_rec_c):
                    lg.grad = alpha * gc + beta * lam_c * gr
            else:
                # one adaptive weight (continuous block only)
                g_cls = torch.autograd.grad(loss_cls_val, X_num, retain_graph=True)[0]
                g_rec = torch.autograd.grad(loss_rec_raw, X_num, retain_graph=True)[0]
                lam = (g_cls.norm() / (g_rec.norm() + 1e-8)).clamp(max=1e4).detach()
                ((alpha * loss_cls_val) + (beta * lam * loss_rec_raw)).backward()

            optimizer.step()

            with torch.no_grad():
                _lo = torch.as_tensor(self._cont_lo, dtype=X_num.dtype, device=X_num.device)
                _hi = torch.as_tensor(self._cont_hi, dtype=X_num.dtype, device=X_num.device)
                X_num.data.copy_(torch.max(torch.min(X_num.data, _hi), _lo))

        # refined samples after the last step: continuous as is, categorical = argmax category
        X_post = np.zeros((N, D), dtype=np.float32)
        X_num_np = X_num.detach().cpu().numpy()
        for new_i, orig_i in enumerate(num_idx):
            X_post[:, orig_i] = X_num_np[:, new_i]
        for logits, (col_pos, allowed_np, _) in zip(cat_logits_list, cat_meta):
            X_post[:, col_pos] = allowed_np[logits.detach().argmax(dim=1).cpu().numpy()]

        X_refined_df = pd.DataFrame(X_post, columns=X_gen_df.columns)
        return snap_categoricals(X_refined_df, cat_allowed)  # categorical columns float32 -> int
