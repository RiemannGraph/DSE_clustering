import numpy as np
import torch
from sklearn import metrics
from munkres import Munkres
import networkx as nx


def decoding_cluster_from_tree(manifold, tree: nx.Graph, num_clusters, num_nodes, height):
    root = tree.nodes[num_nodes]
    root_coords = root['coords']
    dist_dict = {}  # for every height of tree
    for u in tree.nodes():
        if u != num_nodes:  # u is not root
            h = tree.nodes[u]['height']
            dist_dict[h] = dist_dict.get(h, {})
            dist_dict[h].update({u: manifold.dist(root_coords, tree.nodes[u]['coords']).numpy()})

    h = 1
    sorted_dist_list = sorted(dist_dict[h].items(), reverse=False, key=lambda x: x[1])
    count = len(sorted_dist_list)
    group_list = [([u], dist) for u, dist in sorted_dist_list]  # [ ([u], dist_u) ]
    while len(group_list) <= 1:
        h = h + 1
        sorted_dist_list = sorted(dist_dict[h].items(), reverse=False, key=lambda x: x[1])
        count = len(sorted_dist_list)
        group_list = [([u], dist) for u, dist in sorted_dist_list]

    while count > num_clusters:
        group_list, count = merge_nodes_once(manifold, root_coords, tree, group_list, count)

    while count < num_clusters and h <= height:
        h = h + 1   # search next level
        pos = 0
        while pos < len(group_list):
            v1, d1 = group_list[pos]  # node to split
            sub_level_set = []
            v1_coord = tree.nodes[v1[0]]['coords']
            for u, v in tree.edges(v1[0]):
                if tree.nodes[v]['height'] == h:
                    v_coords = tree.nodes[v]['coords']
                    dist = manifold.dist(v_coords, v1_coord).cpu().numpy()
                    sub_level_set.append(([v], dist))    # [ ([v], dist_v) ]
            if len(sub_level_set) <= 1:
                pos += 1
                continue
            sub_level_set = sorted(sub_level_set, reverse=False, key=lambda x: x[1])
            count += len(sub_level_set) - 1
            if count > num_clusters:
                while count > num_clusters:
                    sub_level_set, count = merge_nodes_once(manifold, v1_coord, tree, sub_level_set, count)
                del group_list[pos]  # del the position node which will be split
                group_list += sub_level_set    # Now count == num_clusters
                break
            elif count == num_clusters:
                del group_list[pos]  # del the position node which will be split
                group_list += sub_level_set
                break
            else:
                del group_list[pos]
                group_list += sub_level_set
                pos += 1

    cluster_dist = {}
    for i in range(len(group_list)):
        u_list, _ = group_list[i]
        group = []
        for u in u_list:
            index = tree.nodes[u]['children'].tolist()
            group += index
        cluster_dist.update({k: i for k in group})
    results = sorted(cluster_dist.items(), key=lambda x: x[0])
    results = np.array([x[1] for x in results])
    return results


def merge_nodes_once(manifold, root_coords, tree, group_list, count):
    # group_list should be ordered ascend
    v1, v2 = group_list[-1], group_list[-2]
    merged_node = v1[0] + v2[0]
    merged_coords = torch.stack([tree.nodes[v]['coords'] for v in merged_node], dim=0)
    merged_point = manifold.frechet_mean(merged_coords)
    merged_dist = manifold.dist(merged_point, root_coords).cpu().numpy()
    merged_item = (merged_node, merged_dist)
    del group_list[-2:]
    group_list.append(merged_item)
    group_list = sorted(group_list, reverse=False, key=lambda x: x[1])
    count -= 1
    return group_list, count


class cluster_metrics:
    def __init__(self, trues, predicts):
        self.true_label = trues
        self.pred_label = predicts

    def clusterAcc(self):
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        from sklearn import metrics

        true_label = np.array(self.true_label)
        pred_label = np.array(self.pred_label)

        l1 = list(set(true_label))
        l2 = list(set(pred_label))
        numclass1 = len(l1)
        numclass2 = len(l2)

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            for j, c2 in enumerate(l2):
                cost[i, j] = np.sum((true_label == c1) & (pred_label == c2))

        row_ind, col_ind = linear_sum_assignment(-cost)

        pred_to_true = {}
        for r, c in zip(row_ind, col_ind):
            pred_to_true[l2[c]] = l1[r]

        if len(pred_to_true) < len(l2):
            fallback_class = true_label[np.bincount(true_label).argmax()]
            for c2 in l2:
                if c2 not in pred_to_true:
                    pred_to_true[c2] = fallback_class

        new_predict = np.array([pred_to_true[pred] for pred in pred_label])

        acc = metrics.accuracy_score(true_label, new_predict)

        return acc

    def evaluateFromLabel(self, use_acc=False):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        if use_acc:
            acc = self.clusterAcc()
            return acc, nmi, adjscore
        else:
            return nmi, adjscore


def cal_AUC_AP(scores, trues):
    auc = metrics.roc_auc_score(trues, scores)
    ap = metrics.average_precision_score(trues, scores)
    return auc, ap