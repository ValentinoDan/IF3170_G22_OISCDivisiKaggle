import heapq
import pickle
import numpy as np
import pandas as pd

class DecisionTreeNode:
    def __init__(
        self,
        feature_idx=None,
        feature_name=None,
        is_categorical=False,
        threshold=None,
        category_values=None,
        left=None,
        right=None,
        missing_direction="left",
        value=None,
        probabilities=None,
        n_samples=0,
        impurity=0.0,
    ):
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.is_categorical = is_categorical
        self.threshold = threshold
        self.category_values = category_values
        self.left = left
        self.right = right
        self.missing_direction = missing_direction
        self.value = value
        self.probabilities = probabilities
        self.n_samples = n_samples
        self.impurity = impurity

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
        max_leaf_nodes=None,
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

        if max_leaf_nodes is not None and (
            not isinstance(max_leaf_nodes, int) or max_leaf_nodes < 2
        ):
            raise ValueError(f"max_leaf_nodes must be an integer >= 2 or None, got {max_leaf_nodes}")

        self.validate_max_features(max_features)

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.ccp_alpha = float(ccp_alpha)
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes

        self.root = None
        self.classes_ = None
        self.n_classes_ = 0
        self.n_features_in_ = 0
        self.feature_names_ = None
        self.feature_types_ = None

    @staticmethod
    def validate_max_features(val):
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

    def resolve_max_features(self, n_features):
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
    def gini(counts, total):
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
        self.resolve_max_features(self.n_features_in_)

        self.classes_, y_encoded = np.unique(y_arr, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError(f"Classifier can't train when only one class is present. Classes found: {self.classes_}")

        rng = np.random.RandomState(self.random_state)
        indices = np.arange(X_arr.shape[0])

        if self.max_leaf_nodes is None:
            self.root = self.build_tree(X_arr, y_encoded, indices, depth=0, rng=rng)
        else:
            self.root = self.build_tree_best_first(X_arr, y_encoded, indices, rng)

        if self.ccp_alpha > 0.0:
            self.prune_cost_complexity(X_arr.shape[0])

        return self

    def build_tree(self, X, y, indices, depth, rng):
        n_samples = len(indices)
        node_counts = np.bincount(y[indices], minlength=self.n_classes_)
        node_probs = node_counts / n_samples
        majority_class = int(np.argmax(node_counts))
        node_impurity = self.gini(node_counts, n_samples)

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

        best_split = self.find_best_split(X, y, indices, rng)

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

        left_child = self.build_tree(X, y, left_idx, depth + 1, rng)
        right_child = self.build_tree(X, y, right_idx, depth + 1, rng)

        return DecisionTreeNode(
            feature_idx=best_split["feature_idx"],
            feature_name=self.feature_names_[best_split["feature_idx"]],
            is_categorical=best_split["is_categorical"],
            threshold=best_split["threshold"],
            category_values=best_split["category_values"],
            left=left_child,
            right=right_child,
            missing_direction=best_split["missing_direction"],
            probabilities=node_probs,
            n_samples=n_samples,
            impurity=node_impurity,
        )

    def make_leaf(self, y, indices):
        counts = np.bincount(y[indices], minlength=self.n_classes_)
        n_samples = len(indices)
        return DecisionTreeNode(
            value=int(np.argmax(counts)),
            probabilities=counts / n_samples,
            n_samples=n_samples,
            impurity=self.gini(counts, n_samples),
        )

    # Best-first growth: Shi, H. (2007), Best-first Decision Tree Learning.
    def build_tree_best_first(self, X, y, indices, rng):
        """Expand the leaf with the largest dataset-weighted Gini reduction."""
        root = self.make_leaf(y, indices)
        heap = []
        sequence = 0

        def queue_split(node, node_indices, depth):
            nonlocal sequence
            if (
                (self.max_depth is not None and depth >= self.max_depth)
                or node.n_samples < self.min_samples_split
                or node.impurity == 0.0
            ):
                return

            split = self.find_best_split(X, y, node_indices, rng)
            if split is None or split["gain"] < self.min_impurity_decrease:
                return

            weighted_gain = (node.n_samples / len(indices)) * split["gain"]
            heapq.heappush(heap, (-weighted_gain, sequence, node, node_indices, depth, split))
            sequence += 1

        queue_split(root, indices, 0)
        leaf_count = 1

        while heap and leaf_count < self.max_leaf_nodes:
            _, _, node, node_indices, depth, split = heapq.heappop(heap)
            left_indices = split["left_indices"]
            right_indices = split["right_indices"]

            if (
                len(left_indices) < self.min_samples_leaf
                or len(right_indices) < self.min_samples_leaf
            ):
                continue

            node.value = None
            node.feature_idx = split["feature_idx"]
            node.feature_name = self.feature_names_[split["feature_idx"]]
            node.is_categorical = split["is_categorical"]
            node.threshold = split["threshold"]
            node.category_values = split["category_values"]
            node.left = self.make_leaf(y, left_indices)
            node.right = self.make_leaf(y, right_indices)
            node.missing_direction = split["missing_direction"]
            leaf_count += 1

            queue_split(node.left, left_indices, depth + 1)
            queue_split(node.right, right_indices, depth + 1)

        return root

    def find_best_split(self, X, y, indices, rng):
        n_features = X.shape[1]
        n_samples = len(indices)
        node_counts = np.bincount(y[indices], minlength=self.n_classes_)
        node_impurity = self.gini(node_counts, n_samples)
        k_features = self.resolve_max_features(n_features)

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
            if n_valid < 2:
                continue

            valid_indices = indices[valid_mask]
            missing_indices = indices[missing_mask]
            n_miss = len(missing_indices)
            y_valid = y[valid_indices]
            valid_counts = np.bincount(y_valid, minlength=self.n_classes_)
            missing_counts = node_counts - valid_counts

            if is_cat:
                unique_vals = pd.unique(feat_vals[valid_mask])
                if len(unique_vals) <= 1:
                    continue
                if self.n_classes_ == 2:
                    # Ordered categorical Gini splits (Breiman et al., 1984, CART).
                    category_rates = []
                    valid_categories = feat_vals[valid_mask]
                    for category in unique_vals:
                        category_rows = valid_categories == category
                        category_rates.append(
                            np.mean(y_valid[category_rows] == 1)
                        )
                    ordered_categories = unique_vals[
                        np.argsort(category_rates, kind="stable")
                    ]
                    candidates = [
                        (None, tuple(ordered_categories[:cut]))
                        for cut in range(1, len(ordered_categories))
                    ]
                else:
                    candidates = [(category, None) for category in unique_vals]

                for thresh, category_values in candidates:
                    if category_values is None:
                        left_mask_valid = feat_vals[valid_mask] == thresh
                    else:
                        left_mask_valid = np.isin(
                            feat_vals[valid_mask], category_values
                        )

                    n_left = int(np.sum(left_mask_valid))
                    n_right = n_valid - n_left
                    if n_left == 0 or n_right == 0:
                        continue

                    left_counts = np.bincount(y_valid[left_mask_valid], minlength=self.n_classes_)
                    directions = ("left", "right") if n_miss > 0 else ("left",)

                    # Missing-direction search: Chen & Guestrin (2016), KDD '16, Algorithm 3.
                    # Split scoring: CART Gini (Breiman et al., 1984).
                    for direction in directions:
                        if direction == "left" and n_miss > 0:
                            candidate_left_counts = left_counts + missing_counts
                            candidate_n_left = n_left + n_miss
                        else:
                            candidate_left_counts = left_counts
                            candidate_n_left = n_left

                        candidate_right_counts = node_counts - candidate_left_counts
                        candidate_n_right = n_samples - candidate_n_left
                        if (
                            candidate_n_left < self.min_samples_leaf
                            or candidate_n_right < self.min_samples_leaf
                        ):
                            continue

                        left_imp = self.gini(candidate_left_counts, candidate_n_left)
                        right_imp = self.gini(candidate_right_counts, candidate_n_right)
                        gain = node_impurity - (
                            (candidate_n_left / n_samples) * left_imp
                            + (candidate_n_right / n_samples) * right_imp
                        )

                        if gain > best_gain:
                            best_gain = gain
                            missing_dir = (
                                direction
                                if n_miss > 0
                                else ("left" if n_left >= n_right else "right")
                            )
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
                                "category_values": category_values,
                                "gain": gain,
                                "left_indices": left_subset,
                                "right_indices": right_subset,
                                "missing_direction": missing_dir,
                            }
            else:
                valid_num = feat_vals_num[valid_mask]
                sorted_order = np.argsort(valid_num, kind="stable")
                sorted_num = valid_num[sorted_order]
                sorted_y = y_valid[sorted_order]
                unique_vals, group_counts = np.unique(sorted_num, return_counts=True)
                if len(unique_vals) <= 1:
                    continue

                group_ends = np.cumsum(group_counts)
                left_sizes = group_ends[:-1]
                right_sizes = n_valid - left_sizes
                cumulative_class_counts = np.cumsum(
                    np.eye(self.n_classes_, dtype=np.int64)[sorted_y], axis=0
                )
                left_counts = cumulative_class_counts[group_ends[:-1] - 1]
                directions = ("left", "right") if n_miss > 0 else ("left",)

                for direction in directions:
                    if direction == "left" and n_miss > 0:
                        candidate_left_counts = left_counts + missing_counts
                        candidate_n_left = left_sizes + n_miss
                    else:
                        candidate_left_counts = left_counts
                        candidate_n_left = left_sizes

                    candidate_right_counts = node_counts - candidate_left_counts
                    candidate_n_right = n_samples - candidate_n_left
                    valid_boundaries = (
                        (candidate_n_left >= self.min_samples_leaf)
                        & (candidate_n_right >= self.min_samples_leaf)
                    )
                    if not np.any(valid_boundaries):
                        continue

                    left_impurities = 1.0 - np.sum(
                        (candidate_left_counts / candidate_n_left[:, None]) ** 2,
                        axis=1,
                    )
                    right_impurities = 1.0 - np.sum(
                        (candidate_right_counts / candidate_n_right[:, None]) ** 2,
                        axis=1,
                    )
                    gains = node_impurity - (
                        (candidate_n_left / n_samples) * left_impurities
                        + (candidate_n_right / n_samples) * right_impurities
                    )
                    gains[~valid_boundaries] = -np.inf
                    best_boundary = int(np.argmax(gains))
                    gain = float(gains[best_boundary])

                    if gain > best_gain:
                        left_value = unique_vals[best_boundary]
                        right_value = unique_vals[best_boundary + 1]
                        with np.errstate(over="ignore", invalid="ignore"):
                            threshold = left_value + (right_value - left_value) / 2.0
                        if (
                            not np.isfinite(threshold)
                            or threshold < left_value
                            or threshold >= right_value
                        ):
                            threshold = left_value

                        left_mask_valid = valid_num <= threshold
                        n_left = int(left_sizes[best_boundary])
                        n_right = int(right_sizes[best_boundary])
                        best_gain = gain
                        missing_dir = (
                            direction
                            if n_miss > 0
                            else ("left" if n_left >= n_right else "right")
                        )
                        left_subset = valid_indices[left_mask_valid]
                        right_subset = valid_indices[~left_mask_valid]

                        if n_miss > 0:
                            if missing_dir == "left":
                                left_subset = np.concatenate([left_subset, missing_indices])
                            else:
                                right_subset = np.concatenate([right_subset, missing_indices])

                        best_split = {
                            "feature_idx": feat_idx,
                            "is_categorical": False,
                            "threshold": threshold,
                            "category_values": None,
                            "gain": gain,
                            "left_indices": left_subset,
                            "right_indices": right_subset,
                            "missing_direction": missing_dir,
                        }

        return best_split

    def prepare_X_for_predict(self, X):
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
        X_arr = self.prepare_X_for_predict(X)
        n_samples = X_arr.shape[0]
        probs = np.zeros((n_samples, self.n_classes_), dtype=float)

        for i in range(n_samples):
            probs[i] = self.predict_row_proba(self.root, X_arr[i])

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return self.classes_[pred_idx]

    def predict_row_proba(self, node, row):
        if node.is_leaf:
            return node.probabilities

        val = row[node.feature_idx]

        if pd.isna(val):
            next_node = node.left if node.missing_direction == "left" else node.right
            return self.predict_row_proba(next_node, row)

        if node.is_categorical:
            category_values = getattr(node, "category_values", None)
            if category_values is None:
                is_left = val == node.threshold
            else:
                is_left = val in category_values
        else:
            try:
                is_left = (float(val) <= float(node.threshold))
            except (ValueError, TypeError):
                is_left = (node.missing_direction == "left")

        next_node = node.left if is_left else node.right
        return self.predict_row_proba(next_node, row)

    def prune_cost_complexity(self, n_total):
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