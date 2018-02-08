from math import log
from sys import maxsize
from collections import Counter

import pandas as pd


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
        return self.value


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
            node["children"].append({
                "path_value": list(self.path_values_list[index]),
                "node": self.children[index].structure()
            })

        return node

    def path(self, value):
        print("\tPredicted Value {}".format(value))
        for index in range(len(self.path_list)):
            if isinstance(self.path_list[index], tuple):
                if index == 0:
                    if value < self.path_list[index][0]:
                        print("\tPath: {}".format(self.path_list[index]))
                        print("\t\tTuple Path")
                        print("\t\t\tLeft Branch")
                        return self.children[index]
                if index == 1:
                    if self.path_list[index][1] > value > self.path_list[index][0]:
                        print("\tPath: {}".format(self.path_list[index]))
                        print("\t\tTuple Path")
                        print("\t\t\tMiddle Branch")
                        return self.children[index]
                if index == 2:
                    if value > self.path_list[index][0]:
                        print("\tPath: {}".format(self.path_list[index]))
                        print("\t\tTuple Path")
                        print("\t\t\tRight Branch")
                        return self.children[index]
            else:
                print("\t\tNon-Tuple Path")
                if value in self.path_list:
                    print("\t\t\tValue exists in node")
                    return self.children[self.path_list.index(value)]
                else:
                    print("\t\t\tValue not found in node")
                    class_count = Counter(self.classes)
                    return DecisionTreeLeaf(class_count.most_common(1))


class DecisionTree:
    def __init__(self, train, target):
        indexes = [str(index) for index in range(len(train))]
        train.set_index([indexes])
        self.root = DecisionTreeNode(None, target)
        self.root.columns = list(train.columns)
        self.create_tree(train, target, self.root)

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
        print(parent.path_values_list)
        parent.path_list = list(nodes[lowest_key]["row_class_pair"].keys())
        print(parent.path_list)

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


class DecisionTreeClassifier:
    def __init__(self, train, target):
        self.tree = DecisionTree(train, target)

    def structure(self):
        return self.tree.structure()

    def predict(self, data):
        print("Start Predict")
        print(data.columns)
        print(self.tree.root.columns)

        if len(data.columns) != len(self.tree.root.columns):
            return None

        node = self.tree.root

        print(isinstance(node, DecisionTreeLeaf))

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
