import matplotlib.pyplot as plt

def plot_decision_tree(
    model,
    feature_names=None,
    class_names=None,
    max_depth=None,
    top_n=None,
    figsize=None,
    ax=None,
    show=False,
    show_probabilities=False,
):
    """Plot a fitted tree; top_n limits visible split nodes in breadth-first order."""
    root = getattr(model, "root", None)
    if root is None:
        raise ValueError("model must be a fitted DecisionTree.")
    if max_depth is not None and (
        not isinstance(max_depth, int) or max_depth < 0
    ):
        raise ValueError("max_depth must be a non-negative integer or None.")
    if top_n is not None and (
        not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 0
    ):
        raise ValueError("top_n must be a non-negative integer or None.")

    if feature_names is None:
        feature_names = getattr(model, "feature_names_", None)
    if feature_names is not None:
        feature_names = list(feature_names)

    if class_names is None:
        class_names = getattr(model, "classes_", None)
    elif isinstance(class_names, str):
        raise TypeError("class_names must be a sequence, not a string.")
    if class_names is not None:
        class_names = list(class_names)

    positions = {}
    truncated = set()
    edges = []
    cursor = 0
    deepest_level = 0

    def is_leaf(node):
        return bool(getattr(node, "is_leaf", getattr(node, "value", None) is not None))

    visible_split_ids = None
    if top_n is not None:
        visible_split_ids = set()
        queue = [(root, 0)]
        queue_index = 0
        while queue_index < len(queue) and len(visible_split_ids) < top_n:
            node, depth = queue[queue_index]
            queue_index += 1
            if is_leaf(node) or (max_depth is not None and depth >= max_depth):
                continue

            visible_split_ids.add(id(node))
            left = getattr(node, "left", None)
            right = getattr(node, "right", None)
            if left is not None:
                queue.append((left, depth + 1))
            if right is not None:
                queue.append((right, depth + 1))

    def layout(node, depth):
        nonlocal cursor, deepest_level
        deepest_level = max(deepest_level, depth)
        left = getattr(node, "left", None)
        right = getattr(node, "right", None)
        clipped = (
            not is_leaf(node)
            and (
                (max_depth is not None and depth >= max_depth)
                or (
                    visible_split_ids is not None
                    and id(node) not in visible_split_ids
                )
            )
        )

        if is_leaf(node) or clipped or (left is None and right is None):
            x = cursor
            cursor += 1
            positions[id(node)] = (x, -depth)
            if clipped:
                truncated.add(id(node))
            return x

        child_x = []
        if left is not None:
            child_x.append(layout(left, depth + 1))
        if right is not None:
            child_x.append(layout(right, depth + 1))
        x = sum(child_x) / len(child_x)
        positions[id(node)] = (x, -depth)

        if left is not None:
            edges.append((node, left, "left"))
        if right is not None:
            edges.append((node, right, "right"))
        return x

    layout(root, 0)

    def feature_label(node):
        index = getattr(node, "feature_idx", None)
        if feature_names is not None and index is not None and index < len(feature_names):
            return str(feature_names[index])
        name = getattr(node, "feature_name", None)
        return str(name) if name is not None else f"x{index}"

    def class_label(node):
        value = getattr(node, "value", None)
        if class_names is not None and value is not None:
            try:
                return str(class_names[int(value)])
            except (IndexError, TypeError, ValueError):
                pass
        return str(value)

    def format_impurity(node):
        impurity = getattr(node, "impurity", None)
        return "" if impurity is None else f"Gini: {float(impurity):.3f}"

    def node_text(node):
        if id(node) in truncated:
            return "…\nsubtree hidden"

        parts = []
        if is_leaf(node):
            parts.append(f"class: {class_label(node)}")
        else:
            parts.append(feature_label(node))
        parts.append(f"samples: {getattr(node, 'n_samples', 0)}")
        impurity = format_impurity(node)
        if impurity:
            parts.append(impurity)

        probabilities = getattr(node, "probabilities", None)
        if show_probabilities and probabilities is not None:
            parts.append(
                "p: [" + ", ".join(f"{float(p):.2f}" for p in probabilities) + "]"
            )
        return "\n".join(parts)

    def edge_text(node, side):
        threshold = getattr(node, "threshold", None)
        if getattr(node, "is_categorical", False):
            category_values = getattr(node, "category_values", None)
            if category_values is not None:
                value = "{" + ", ".join(map(str, category_values)) + "}"
                if len(value) > 24:
                    value = value[:21] + "…}"
                condition = (
                    f"in {value}" if side == "left" else f"not in {value}"
                )
            else:
                value = str(threshold)
                if len(value) > 18:
                    value = value[:15] + "…"
                condition = f"= {value}" if side == "left" else f"≠ {value}"
        else:
            try:
                value = f"{float(threshold):,.2f}".rstrip("0").rstrip(".")
            except (TypeError, ValueError):
                value = str(threshold)
            condition = f"≤ {value}" if side == "left" else f"> {value}"

        if getattr(node, "missing_direction", None) == side:
            condition += ", NaN"
        return condition

    if ax is None:
        if figsize is None:
            figsize = (max(8.0, cursor * 3.0), max(4.0, (deepest_level + 1) * 1.5))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for parent, child, side in edges:
        x1, y1 = positions[id(parent)]
        x2, y2 = positions[id(child)]
        ax.plot([x1, x2], [y1 - 0.12, y2 + 0.12], color="#555555", linewidth=1.2, zorder=1)
        label_x = x1 + 0.58 * (x2 - x1)
        label_y = y1 + 0.58 * (y2 - y1)
        ax.text(
            label_x,
            label_y,
            edge_text(parent, side),
            ha="right" if side == "left" else "left",
            va="center",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
            zorder=2,
        )

    visible_nodes = []

    def collect_visible(node):
        visible_nodes.append(node)
        if id(node) in truncated or is_leaf(node):
            return
        if getattr(node, "left", None) is not None:
            collect_visible(node.left)
        if getattr(node, "right", None) is not None:
            collect_visible(node.right)

    collect_visible(root)
    for node in visible_nodes:
        x, y = positions[id(node)]
        facecolor = "#EAF1F8" if not is_leaf(node) and id(node) not in truncated else "#F3F4F6"
        ax.text(
            x,
            y,
            node_text(node),
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": facecolor, "edgecolor": "#444444", "linewidth": 1.0},
            zorder=3,
        )

    ax.set_title("Decision Tree")
    ax.set_xlim(-0.7, max(cursor - 0.3, 0.7))
    ax.set_ylim(-deepest_level - 0.7, 0.7)
    ax.axis("off")
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax