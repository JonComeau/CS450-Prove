from math import log
from sys import maxsize
from collections import Counter

import pandas as pd


# Taken from https://stackoverflow.com/a/2135920/7747350
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


class DecisionTreeLeaf:
    def __init__(self, value):
        self.class_value = value

    def structure(self):
        return {
            "value": self.class_value
        }

    def path(self, value):
        return self.class_value


class DecisionTreeNode:
    def __init__(self, parent, classes):
        self.parent = parent
        self.classes = classes
        self.attribute = ""
        self.columns = []
        self.children = []
        self.path_values_list = []
        self.path_list = []

    def structure(self):
        node = {
            "classes": list(self.classes),
            "column": self.attribute,
            "children": []
        }

        for index in range(len(self.children)):
            is_child_leaf = isinstance(self.children[index], DecisionTreeLeaf)

            node["children"].append({
                "path_value": list(self.path_list[index]) if isinstance(self.path_list[index], tuple) else self.path_list[index],
                ("node" if not is_child_leaf else "leaf"): self.children[index].structure()
            })

        return node

    def path(self, value, proba=False):
        print("\tPredicted Value {}".format(value))

        if value in self.path_list:
            print("\t\tNon-Tuple Path")
            print("\t\t\tValue exists in node")
            if proba:
                return self.children[self.path_list.index(value)], self.path_list[self.path_list.index(value)]
            else:
                return self.children[self.path_list.index(value)]

        for index in range(len(self.path_list)):
            if isinstance(self.path_list[index], tuple) and (index == 0 and value < self.path_list[index][0]) or (index == 1 and self.path_list[index][1] > value > self.path_list[index][0]) or (index == 2 and value > self.path_list[index][0]):
                print("\tPath: {}".format(self.path_list[index]))
                print("\t\tTuple Path")
                print("\t\t\tLeft Branch")
                if proba:
                    print("\t\t\t\tProbability Included")
                    return self.children[index], self.path_list[index]
                else:
                    return self.children[index]

        print("\t\t\tValue not found in node")
        class_count = Counter(self.classes)
        return DecisionTreeLeaf(class_count.most_common(1))


class DecisionTree:
    def __init__(self, train, target, prune = False):
        indexes = [str(index) for index in range(len(train))]
        train.set_index([indexes])
        self.root = DecisionTreeNode(None, target)
        self.root.columns = list(train.columns)
        self.create_tree(train, target, self.root)

        if prune:
            self.prune()

    def create_tree(self, train, target, parent):
        target = [temp for temp in target]

        nodes = {}

        for name, values in train.iteritems():
            entropies = {}
            unique = list(set(values))
            list.sort(unique)

            if len(unique) > 3:
                unique = list(split(unique, 3))
                cont = True
            else:
                cont = False

            row_class_pair = {}

            if cont:
                row_class_pair[(min(unique[1]),)] = []
                row_class_pair[(min(unique[1]), max(unique[1]))] = []
                row_class_pair[(max(unique[1]),)] = []
            else:
                row_class_pair = {value: [] for value in unique}

            indexes = values.index.values.tolist()

            row_class_pair_keys = list(row_class_pair.keys())

            for index in range(len(indexes)):
                for ind in range(len(row_class_pair_keys)):
                    loc_value = values.loc[indexes[index]]
                    key = row_class_pair_keys[ind]

                    if cont:
                        if ind == 0:
                            if key[0] > loc_value:
                                row_class_pair[key].append({
                                    "line": index,
                                    "class": target[index]
                                })
                                break
                        elif ind == 1:
                            key_min, key_max = key
                            if (loc_value >= key_min) & (loc_value <= key_max):
                                row_class_pair[key].append({
                                    "line": index,
                                    "class": target[index]
                                })
                                break
                        elif ind == 2:
                            if key[0] < loc_value:
                                row_class_pair[key].append({
                                    "line": index,
                                    "class": target[index]
                                })
                                break
                    else:
                        if loc_value == key:
                            row_class_pair[key].append({
                                "line": index,
                                "class": target[index]
                            })

            for key, value in row_class_pair.items():
                classes = [val["class"] for val in value]
                entropies[key] = calc_entropy(classes)

            total_entropy = 0

            for key, items in row_class_pair.items():
                total_entropy += (entropies[key] * len(items))

            nodes[name] = {
                "row_class_pair": row_class_pair,
                "entropy": total_entropy / len(target)
            }

        lowest_entropy = maxsize
        lowest_key = ""

        for key, value in nodes.items():
            if value["entropy"] < lowest_entropy:
                lowest_entropy = value["entropy"]
                lowest_key = key

        del train[lowest_key]

        parent.attribute = lowest_key
        parent.path_values_list = [nodes[lowest_key]["row_class_pair"][key] for key in nodes[lowest_key]["row_class_pair"].keys()]
        parent.path_list = list(nodes[lowest_key]["row_class_pair"].keys())

        for ind in range(len(parent.path_values_list)):
            lines = [val["line"] for val in parent.path_values_list[ind]]
            classes = [val["class"] for val in parent.path_values_list[ind]]

            if len(lines) == 0:
                continue

            if train.empty:
                classes_count = {cls: 0 for cls in set(classes)}

                for cls in classes:
                    classes_count[cls] += 1

                max_class_count = 0
                max_class = ""
                splt = False
                splt_classes = []

                for key, value in classes_count.items():
                    if value > max_class_count:
                        splt = False
                        max_class_count = value
                        max_class = key
                    if value == max_class_count:
                        splt = True
                        max_class_count = value
                        splt_classes.append(max_class)
                        splt_classes.append(key)

                if splt:
                    max_classes = list(set(splt_classes))
                    max_classes.sort()
                    parent.children.append(DecisionTreeLeaf(max_classes[0]))
                else:
                    parent.childen.append(DecisionTreeLeaf(max_class))

                continue

            if len(set(classes)) == 1:
                parent.children.append(DecisionTreeLeaf(classes[0]))
                continue
            elif len(set(classes)) == 0:
                continue

            train_new = train.iloc[lines]

            node = DecisionTreeNode(parent, classes)

            node.columns = list(train.columns)

            self.create_tree(train_new, classes, node)

            parent.children.append(node)

    def structure(self):
        return self.root.structure()


class DTClassifier:
    def __init__(self, train, target):
        self.train = train
        self.target = target
        self.tree = DecisionTree(train, target)

    def structure(self):
        return self.tree.structure()

    def predict(self, data):
        print("Start Predict\n")
        print("Columns:\n\t{}".format(", ".join(self.tree.root.columns)))

        if len(data.columns) != len(self.tree.root.columns):
            return None

        node = self.tree.root

        while not isinstance(node, DecisionTreeLeaf):
            column_index = node.columns.index(node.attribute)
            print("Node Attribute: {}".format(node.attribute))
            data.drop(node.attribute, axis=1)
            node = node.path(list(data.iloc[0])[column_index])

        return node.class_value


def calc_entropy(class_list):
    dict = {}
    entropy = 0

    for item in class_list:
        if item in dict.keys():
            dict[item] += 1
        else:
            dict[item] = 1

    values = [value for (key, value) in dict.items()]

    for value in values:
        entropy -= (value / len(class_list)) * log((value / len(class_list)), 2)

    return entropy
