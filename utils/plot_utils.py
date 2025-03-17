import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE

def mobius_add(x, y):
    """Mobius addition in numpy."""
    xy = np.sum(x * y, 1, keepdims=True)
    x2 = np.sum(x * x, 1, keepdims=True)
    y2 = np.sum(y * y, 1, keepdims=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    den = 1 + 2 * xy + x2 * y2
    return num / den


def mobius_mul(x, t):
    """Mobius multiplication in numpy."""
    normx = np.sqrt(np.sum(x * x, 1, keepdims=True))
    return np.tanh(t * np.arctanh(normx)) * x / normx


def geodesic_fn(x, y, nb_points=100):
    """Get coordinates of points on the geodesic between x and y."""
    t = np.linspace(0, 1, nb_points)
    x_rep = np.repeat(x.reshape((1, -1)), len(t), 0)
    y_rep = np.repeat(y.reshape((1, -1)), len(t), 0)
    t1 = mobius_add(-x_rep, y_rep)
    t2 = mobius_mul(t1, t.reshape((-1, 1)))
    return mobius_add(x_rep, t2)


def plot_geodesic(x, y, ax):
    """Plots geodesic between x and y."""
    points = geodesic_fn(x, y)
    ax.plot(points[:, 0], points[:, 1], color='black', linewidth=0.3, alpha=1.)


def plot_leaves(tree, manifold, embeddings, labels, height, save_path=None, colors_dict=None):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    circle = plt.Circle((0, 0), 1.0, color='y', alpha=0.1)
    ax.add_artist(circle)
    for k in range(1, height + 1):
        circle_k = plt.Circle((0, 0), k / (height + 1), color='b', alpha=0.05)
        ax.add_artist(circle_k)
    n = embeddings.shape[0]
    colors_dict = get_colors(labels, color_seed=1234) if colors_dict is None else colors_dict
    colors = [colors_dict[k] for k in labels]
    embeddings = manifold.to_poincare(embeddings).numpy()
    scatter = ax.scatter(embeddings[:n, 0], embeddings[:n, 1], c=colors, s=80, alpha=1.0)
    # legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    # ax.add_artist(legend)
    # ax.scatter(np.array([0]), np.array([0]), c='black')
    for u, v in tree.edges():
        x = manifold.to_poincare(tree.nodes[u]['coords']).numpy()
        y = manifold.to_poincare(tree.nodes[v]['coords']).numpy()
        if tree.nodes[u]['is_leaf'] is False:
            c = 'black' if tree.nodes[u]['height'] == 0 else 'red'
            m = '*' if tree.nodes[u]['height'] == 0 else 's'
            ax.scatter(x[0], x[1], c=c, s=30, marker=m)
        if tree.nodes[v]['is_leaf'] is False:
            c = 'black' if tree.nodes[v]['height'] == 0 else 'red'
            m = '*' if tree.nodes[u]['height'] == 0 else 's'
            ax.scatter(y[0], y[1], c=c, s=30, marker=m)
        plot_geodesic(y, x, ax)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=500)
    plt.show()
    return ax, colors_dict


def get_colors(y, color_seed=1234):
    """random color assignment for label classes."""
    np.random.seed(color_seed)
    colors = {}
    for k in np.unique(y):
        r = np.random.random()
        b = np.random.random()
        g = np.random.random()
        colors[k] = (r, g, b)
    return colors


def plot_nx_graph(G: nx.Graph, root, save_path=None):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    pos = graphviz_layout(G, 'twopi')
    nx.draw(G, pos, ax=ax, with_labels=True)
    plt.savefig(save_path)
    plt.show()


def map_labels(true_labels, predicted_labels):
    D = np.zeros((np.unique(true_labels).size, np.unique(predicted_labels).size))
    for i, tl in enumerate(np.unique(true_labels)):
        for j, pl in enumerate(np.unique(predicted_labels)):
            D[i, j] = np.sum((true_labels == tl) & (predicted_labels == pl))

    row_ind, col_ind = linear_sum_assignment(-D)

    label_map = {}
    for i, j in zip(row_ind, col_ind):
        label_map[np.unique(true_labels)[i]] = np.unique(predicted_labels)[j]

    new_true_labels = np.vectorize(label_map.get)(true_labels)
    new_predicted_labels = predicted_labels

    return new_true_labels, new_predicted_labels


def plot_tsne(z, true_labels, predicted_labels, sample_num=7000, output_path="tsne_visualization.pdf"):
    """
    Visualize embedding by t-SNE with true and predicted labels using circles, and save the figures as PDF.
    """
    new_true_labels, new_predicted_labels = map_labels(true_labels, predicted_labels)

    sample_embeds = z
    sample_true_label = new_true_labels
    sample_predict_label = new_predicted_labels

    ts = TSNE(n_components=2, init='pca', random_state=0)
    ts_embeds = ts.fit_transform(sample_embeds)

    x_min, x_max = np.min(ts_embeds, 0), np.max(ts_embeds, 0)
    norm_ts_embeds = (ts_embeds - x_min) / (x_max - x_min)

    custom_colors = ['#ffff4e', '#329932', '#2a2aff', '#ff0f0f', '#8a148a', '#ffae1c', '#bf80bf', '#9a9a9a']

    all_labels = np.unique(np.concatenate([sample_true_label, sample_predict_label]))
    label_color_map = {label: custom_colors[i % len(custom_colors)] for i, label in enumerate(all_labels)}

    fig_true, ax_true = plt.subplots(figsize=(8, 8))
    for label in all_labels:
        class_mask = sample_true_label == label
        ax_true.scatter(norm_ts_embeds[class_mask, 0], norm_ts_embeds[class_mask, 1],
                        color=label_color_map[label], s=10, label=f'Class {label}')
    ax_true.set_title('t-SNE with True Labels', fontsize=14)
    ax_true.set_xticks([])
    ax_true.set_yticks([])
    ax_true.axis('off')

    fig_pred, ax_pred = plt.subplots(figsize=(8, 8))
    for label in all_labels:
        class_mask = sample_predict_label == label
        ax_pred.scatter(norm_ts_embeds[class_mask, 0], norm_ts_embeds[class_mask, 1],
                        color=label_color_map[label], s=10, label=f'Class {label}')
    ax_pred.set_title('t-SNE with Predicted Labels', fontsize=14)
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])
    ax_pred.axis('off')

    fig_true.savefig(output_path.replace(".pdf", "_true.pdf"), bbox_inches='tight')
    fig_pred.savefig(output_path.replace(".pdf", "_pred.pdf"), bbox_inches='tight')
    plt.show()

    return fig_true, fig_pred

