import numpy as np 
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Decision Trees
def dt(dataset):

    # Train the decision tree classifier by fitting the DecisionTreeClassifier class using the datasets.load_iris` dataset:
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
    clf.fit(X_train, y_train)

    # Decision tree structure:
    # The decision classifier has an attribute called ‘tree_’ which allows access
    # to low level attributes such as ‘node_count’, the total number of nodes,
    # and ‘max_depth’ (max depth of the tree). 

    # The tree structure is represented as a number of parallel arrays. The
    # ‘i-th’ element of each array holds information about the node ‘i’. Node 0 is
    # the tree's root. Some of the array features only applies to leaves or split
    # nodes.

    # In this example the arrays feature and threshold only apply to split nodes. 
    # The values for leaf nodes in these arrays are therefore arbitrary.
    #
    # Array definitions:
    #  children_left[i]. Id of the left child of node ‘i’ or -1 if leaf node.
    #  children_right[i].  Id of the right child of node ‘i’ or -1 if leaf node.
    #  feature[i].  Feature used for splitting node ‘i’.
    #  threshold[i]. Threshold value at node ‘i’.
    #  n_node_samples[i]. Number of training samples reaching node ‘i’.
    #  impurity[i]. The impurity at node ‘i’.

    # The tree structure is traversed and array nodes with different properties
    # are computed:

    # Compute the depth of each node and decide whether or not it is a leaf:
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
            node_id, depth = stack.pop()  # Ensures each node is only visited once.

        node_depth[node_id] = depth

        # If the left and right child of a node is not the same, it is  a split node.
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}node={node} is a leaf node.".format(
                space=node_depth[i] * "\t", node=i))
        else:
            print("{space}node={node} is a split node: "
                "go to node {left} if X[:, {feature}] <= {threshold} "
                "else to node {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i]))

    # Plot the decision tree:
    tree.plot_tree(clf)
    plt.title('Decision Tree with array nodes that have different properties.')
    plt.show()
