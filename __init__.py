class MultiModel:
    test_scores = {}
    train_scores = {}
    models = {}

    def __init__(self, n_models, typeof):
        self.n_models = n_models
        self.typeof = typeof

    def load(self, X_train, X_test, y_train, y_test):

        if self.typeof == 'clf':
            # IMPORTING LIBRARIES

            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.neural_network import MLPClassifier
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.naive_bayes import GaussianNB
            from sklearn.svm import SVC
            from sklearn.linear_model import SGDClassifier
            from sklearn.linear_model import LogisticRegression

            self.models = {
                'KNN': KNeighborsClassifier(n_neighbors=1),
                'DecisionTreeClassifiier': DecisionTreeClassifier(max_leaf_nodes=3, random_state=0),
                'RandomForest': RandomForestClassifier(n_estimators=100),
                'MLP': MLPClassifier(activation='logistic', random_state=3),
                'LinearDiscriminant': LinearDiscriminantAnalysis(),
                'GradientBoosting': GradientBoostingClassifier(random_state=0),
                "SVM": SVC(kernel="linear"),
                "Naive_bayes": GaussianNB(),
                'SGDPerceptron': SGDClassifier(loss='perceptron'),
                'LogisticRegression': LogisticRegression(),
                'ExtraTrees': ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=0)
            }


        elif self.typeof == 'regr':
      
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.linear_model import Ridge
            from sklearn.linear_model import Lasso
            from sklearn.tree import DecisionTreeRegressor

            self.models = {
      
                'KNNRegressor': KNeighborsRegressor(n_neighbors=3),
                'LinearRession': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
            }

        modules = list(self.models.keys())
        agg_model = modules[:self.n_models]
        for x in agg_model:
            model = self.models[x]
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            self.test_scores[x] = test_score
            self.train_scores[x] = train_score

        return (self.train_scores, self.test_scores)

    def check(self, accuracy_scores):

        train, test = accuracy_scores

        max_test_score = max(list(test.values()))
        max_train_score = max(list(train.values()))

        max_key_train = [k for k, i in train.items() if i == max_train_score]
        max_key_test = [k for k, i in test.items() if i == max_test_score]

        print(
            'Best train score: {}:{:.2f}%\nBest test score: {}:{:.2f}%'.format(max_key_train, (max_train_score * 100),
                                                                               max_key_test, (max_test_score * 100)))

    def compare(self, accuracy_scores):
        train, test = accuracy_scores

        print('ACCURACY COMPARISON')
        for (model1, accuracy1), (model2, accuracy2) in zip(train.items(), test.items()):
            print('{}:{:.2f}% || {}:{:.2f}%'.format(model1, accuracy1 * 100, model2, accuracy2 * 100))
