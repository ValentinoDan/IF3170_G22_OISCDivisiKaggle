import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DecisionTreeNode:
    def __init__(
        self,
        feature_idx=None,
        feature_name=None,
        is_categorical=False,
        threshold=None,
        left=None,
        right=None,
        missing_direction="left",
        value=None,
        probabilities=None,
        n_samples=0,
        impurity=0.0,
        gain=0.0,
    ):
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.is_categorical = is_categorical
        self.threshold = threshold
        self.left = left
        self.right = right
        self.missing_direction = missing_direction
        self.value = value
        self.probabilities = probabilities
        self.n_samples = n_samples
        self.impurity = impurity
        self.gain = gain

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    "Classification and Regression Tree (CART)"

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        max_features=None,
        random_state=42,
    ):
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth <= 0):
            raise ValueError(f"max_depth must be a positive integer or None, got {max_depth}")

        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError(f"min_samples_split must be integer >= 2, got {min_samples_split}")

        if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be integer >= 1, got {min_samples_leaf}")

        if not isinstance(min_impurity_decrease, (int, float)) or min_impurity_decrease < 0:
            raise ValueError(f"min_impurity_decrease must be >= 0.0, got {min_impurity_decrease}")

        if not isinstance(ccp_alpha, (int, float)) or ccp_alpha < 0:
            raise ValueError(f"ccp_alpha must be >= 0.0, got {ccp_alpha}")

        self._validate_max_features(max_features)

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.ccp_alpha = float(ccp_alpha)
        self.max_features = max_features
        self.random_state = random_state

        self.root = None
        self.classes_ = None
        self.n_classes_ = 0
        self.n_features_in_ = 0
        self.feature_names_ = None
        self.feature_types_ = None

    @staticmethod
    def _validate_max_features(val):
        if val is None:
            return
        if isinstance(val, int):
            if val < 1:
                raise ValueError(f"max_features as int must be >= 1, got {val}")
        elif isinstance(val, float):
            if not (0.0 < val <= 1.0):
                raise ValueError(f"max_features as float must be in (0.0, 1.0], got {val}")
        elif isinstance(val, str):
            if val not in ("sqrt", "log2"):
                raise ValueError(f"Invalid max_features string '{val}'. Valid: 'sqrt', 'log2'")
        else:
            raise ValueError(f"Invalid max_features type: {type(val)}")

    def _resolve_max_features(self, n_features):
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, int):
            if self.max_features > n_features:
                raise ValueError(f"max_features ({self.max_features}) cannot exceed n_features ({n_features})")
            return self.max_features
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        return n_features

    @staticmethod
    def _gini(counts, total):
        if total == 0:
            return 0.0
        probs = counts / total
        return float(1.0 - np.sum(probs ** 2))

    def fit(self, X, y):
        if X is None or y is None:
            raise ValueError("X and y must not be None.")

        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ValueError("Input DataFrame X is empty.")
            self.feature_names_ = list(X.columns)
            self.feature_types_ = [
                not pd.api.types.is_numeric_dtype(X[col]) for col in X.columns
            ]
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)
            if X_arr.size == 0:
                raise ValueError("Input array X is empty.")
            if X_arr.ndim != 2:
                raise ValueError(f"X must be 2-dimensional, got {X_arr.ndim}D.")
            self.feature_names_ = [f"x{i}" for i in range(X_arr.shape[1])]
            self.feature_types_ = [
                not np.issubdtype(X_arr[:, i].dtype, np.number)
                for i in range(X_arr.shape[1])
            ]

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_arr = np.squeeze(y.to_numpy())
        else:
            y_arr = np.squeeze(np.asarray(y))

        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y_arr.shape}")

        if len(y_arr) == 0:
            raise ValueError("Target y is empty.")

        if pd.isna(y_arr).any():
            raise ValueError("Target y contains NaN / missing values.")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"Found input variables with inconsistent numbers of samples: X has {X_arr.shape[0]}, y has {y_arr.shape[0]}"
            )

        self.n_features_in_ = X_arr.shape[1]
        self._resolve_max_features(self.n_features_in_)

        self.classes_, y_encoded = np.unique(y_arr, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError(f"Classifier can't train when only one class is present. Classes found: {self.classes_}")

        rng = np.random.RandomState(self.random_state)
        indices = np.arange(X_arr.shape[0])

        self.root = self._build_tree(X_arr, y_encoded, indices, depth=0, rng=rng)

        if self.ccp_alpha > 0.0:
            self._prune_cost_complexity(X_arr.shape[0])

        return self

    def _build_tree(self, X, y, indices, depth, rng):
        n_samples = len(indices)
        node_counts = np.bincount(y[indices], minlength=self.n_classes_)
        node_probs = node_counts / n_samples
        majority_class = int(np.argmax(node_counts))
        node_impurity = self._gini(node_counts, n_samples)

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or node_impurity == 0.0
            or np.max(node_counts) == n_samples
        ):
            return DecisionTreeNode(
                value=majority_class,
                probabilities=node_probs,
                n_samples=n_samples,
                impurity=node_impurity,
            )

        best_split = self._find_best_split(X, y, indices, rng)

        if best_split is None or best_split["gain"] < self.min_impurity_decrease:
            return DecisionTreeNode(
                value=majority_class,
                probabilities=node_probs,
                n_samples=n_samples,
                impurity=node_impurity,
            )

        left_idx = best_split["left_indices"]
        right_idx = best_split["right_indices"]

        if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
            return DecisionTreeNode(
                value=majority_class,
                probabilities=node_probs,
                n_samples=n_samples,
                impurity=node_impurity,
            )

        left_child = self._build_tree(X, y, left_idx, depth + 1, rng)
        right_child = self._build_tree(X, y, right_idx, depth + 1, rng)

        return DecisionTreeNode(
            feature_idx=best_split["feature_idx"],
            feature_name=self.feature_names_[best_split["feature_idx"]],
            is_categorical=best_split["is_categorical"],
            threshold=best_split["threshold"],
            left=left_child,
            right=right_child,
            missing_direction=best_split["missing_direction"],
            probabilities=node_probs,
            n_samples=n_samples,
            impurity=node_impurity,
            gain=best_split["gain"],
        )

    def _find_best_split(self, X, y, indices, rng):
        n_features = X.shape[1]
        n_samples = len(indices)
        k_features = self._resolve_max_features(n_features)

        if k_features < n_features:
            feature_indices = rng.choice(n_features, size=k_features, replace=False)
        else:
            feature_indices = range(n_features)

        best_gain = -1.0
        best_split = None

        for feat_idx in feature_indices:
            feat_vals = X[indices, feat_idx]
            is_cat = self.feature_types_[feat_idx]

            if is_cat:
                missing_mask = pd.isna(feat_vals)
            else:
                feat_vals_num = feat_vals.astype(float)
                missing_mask = np.isnan(feat_vals_num)

            valid_mask = ~missing_mask
            n_valid = int(np.sum(valid_mask))
            if n_valid < 2 * self.min_samples_leaf:
                continue

            valid_indices = indices[valid_mask]
            missing_indices = indices[missing_mask]
            n_miss = len(missing_indices)
            y_valid = y[valid_indices]
            valid_counts = np.bincount(y_valid, minlength=self.n_classes_)
            valid_impurity = self._gini(valid_counts, n_valid)

            if is_cat:
                unique_vals = pd.unique(feat_vals[valid_mask])
                if len(unique_vals) <= 1:
                    continue
                candidates = unique_vals
            else:
                valid_num = feat_vals_num[valid_mask]
                unique_vals = np.unique(valid_num)
                if len(unique_vals) <= 1:
                    continue
                if len(unique_vals) > 50:
                    candidates = np.percentile(unique_vals, np.linspace(2, 98, 49))
                else:
                    candidates = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in candidates:
                if is_cat:
                    left_mask_valid = (feat_vals[valid_mask] == thresh)
                else:
                    left_mask_valid = (feat_vals_num[valid_mask] <= thresh)

                n_left = int(np.sum(left_mask_valid))
                n_right = n_valid - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                left_counts = np.bincount(y_valid[left_mask_valid], minlength=self.n_classes_)
                right_counts = valid_counts - left_counts

                left_imp = self._gini(left_counts, n_left)
                right_imp = self._gini(right_counts, n_right)

                gain = valid_impurity - ((n_left / n_valid) * left_imp + (n_right / n_valid) * right_imp)
                gain *= (n_valid / n_samples)

                if gain > best_gain:
                    best_gain = gain
                    missing_dir = "left" if n_left >= n_right else "right"
                    left_subset = valid_indices[left_mask_valid]
                    right_subset = valid_indices[~left_mask_valid]

                    if n_miss > 0:
                        if missing_dir == "left":
                            left_subset = np.concatenate([left_subset, missing_indices])
                        else:
                            right_subset = np.concatenate([right_subset, missing_indices])

                    best_split = {
                        "feature_idx": feat_idx,
                        "is_categorical": is_cat,
                        "threshold": thresh,
                        "gain": gain,
                        "left_indices": left_subset,
                        "right_indices": right_subset,
                        "missing_direction": missing_dir,
                    }

        return best_split

    def _prepare_X_for_predict(self, X):
        if self.root is None:
            raise ValueError("This DecisionTree instance is not fitted yet.")

        if X is None:
            raise ValueError("Input X cannot be None.")

        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ValueError("Input DataFrame X is empty.")
            missing_cols = set(self.feature_names_) - set(X.columns)
            if missing_cols:
                raise ValueError(f"X has missing columns required by fitted model: {missing_cols}")
            X_aligned = X[self.feature_names_].to_numpy()
        else:
            X_aligned = np.asarray(X)
            if X_aligned.size == 0:
                raise ValueError("Input array X is empty.")
            if X_aligned.ndim != 2:
                raise ValueError(f"X must be 2-dimensional, got {X_aligned.ndim}D.")
            if X_aligned.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X_aligned.shape[1]} features, but DecisionTree was trained with {self.n_features_in_} features."
                )

        return X_aligned

    def predict_proba(self, X):
        X_arr = self._prepare_X_for_predict(X)
        n_samples = X_arr.shape[0]
        probs = np.zeros((n_samples, self.n_classes_), dtype=float)

        for i in range(n_samples):
            probs[i] = self._predict_row_proba(self.root, X_arr[i])

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return self.classes_[pred_idx]

    def _predict_row_proba(self, node, row):
        if node.is_leaf:
            return node.probabilities

        val = row[node.feature_idx]

        if pd.isna(val):
            next_node = node.left if node.missing_direction == "left" else node.right
            return self._predict_row_proba(next_node, row)

        if node.is_categorical:
            is_left = (val == node.threshold)
        else:
            try:
                is_left = (float(val) <= float(node.threshold))
            except (ValueError, TypeError):
                is_left = (node.missing_direction == "left")

        next_node = node.left if is_left else node.right
        return self._predict_row_proba(next_node, row)

    def _prune_cost_complexity(self, n_total):
        while True:
            internal_nodes = []

            def collect(n):
                if n is None or n.is_leaf:
                    return
                if n.left is not None or n.right is not None:
                    internal_nodes.append(n)
                    collect(n.left)
                    collect(n.right)

            collect(self.root)
            if not internal_nodes:
                break

            min_g = float("inf")
            weakest_node = None

            for node in internal_nodes:
                r_node = (node.n_samples / n_total) * node.impurity

                r_subtree = 0.0
                n_leaves = 0

                def subtree_stats(sub):
                    nonlocal r_subtree, n_leaves
                    if sub is None:
                        return
                    if sub.is_leaf:
                        r_subtree += (sub.n_samples / n_total) * sub.impurity
                        n_leaves += 1
                    else:
                        subtree_stats(sub.left)
                        subtree_stats(sub.right)

                subtree_stats(node)

                if n_leaves <= 1:
                    g = 0.0
                else:
                    g = (r_node - r_subtree) / (n_leaves - 1)

                if g < min_g:
                    min_g = g
                    weakest_node = node

            if min_g <= self.ccp_alpha and weakest_node is not None:
                weakest_node.value = int(np.argmax(weakest_node.probabilities))
                weakest_node.left = None
                weakest_node.right = None
            else:
                break

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)