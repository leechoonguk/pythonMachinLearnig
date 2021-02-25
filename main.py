from sklearn import tree
from sklearn import datasets

data = datasets.load_iris()
dlf = tree.DecisionTreeClassifier()

dlf.fit(data.data,data.target)

print(len(data.target_names[dlf.predict(data.data)]))


if __name__ == '__main__':
    print('test')
