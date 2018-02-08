class KNNClassifier:
    def __init__(self, n_neighbors):
        self.k = n_neighbors

    def fit(self, train_data, train_target):
        self.train_data = train_data
        self.train_target = train_target

    def predict(self, test_data):
        classes = []

        for item in test_data:
            dists = []
            dist_dict = {}

            for point in self.train_data:
                point_dists = []

                for index in range(len(item)):
                    point_dists.append(np.linalg.norm(item[index] - point[index]))

                dists.append(sum(point_dists))

            for index in range(len(dists)):
                # print(str(dists[index]) + ', ')
                dist_dict[dists[index]] = index

            dists.sort()

            neigh_classes = []

            for index in range(self.k):
                ind = dist_dict[dists[index]]

                neigh_classes.append(self.train_target[ind])

            cls_count = dict((cls, 0) for cls in set(self.train_target))

            for cls in neigh_classes:
                cls_count[cls] += 1

            largest_cls = []
            counter = 0

            for key, value in cls_count.items():
                if value > counter:
                    counter = value
                    largest_cls.append(key)

            if len(largest_cls) > 1:
                rand = randint(0, 1)
                largest = largest_cls[rand]
            else:
                largest = largest_cls[0]

            classes.append(largest)

        return classes