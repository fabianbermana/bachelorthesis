
import numpy as np

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


class BlankMLModel:
    """
    A blank forecast model. Used to ensure every subclass has
    all the required functions.
    """

    def __init__(self):
        self.fitted = False
        self.n_predictors = None

    ### -------------------------- ###

    def _check_fit_args(self, X_train, y_train):
        if self.fitted:
            raise Exception('Model is already fitted')
        if len(X_train.shape) != 2:
            raise ValueError('X_train has to be 2D array')
        if len(y_train.shape) != 1:
            raise ValueError('y_train has to be 1D array')
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError('X_train and y_train have different numbers of rows')
        if X_train.shape[1] <= 0:
            raise ValueError('X_train contains no predictors')

    def _check_predict_args(self, X_pred):
        if not self.fitted:
            raise Exception('Model is not fitted')
        if len(X_pred.shape) != 2:
            raise ValueError('X_pred has to be 2D array')
        if X_pred.shape[1] != self.n_predictors:
            raise ValueError('''X_pred does not contain the same number of predictors as
                             training data''')

    ### -------------------------- ###

    def fit(self, X_train, y_train):
        raise NotImplementedError()

    def predict(self, X_pred):
        raise NotImplementedError()


class KNRegressionCV(BlankMLModel):
    """
    Object to produce 1-step ahead forecasts using
    cross-validated k-nearest neighbors regression
    """

    def __init__(self, cv, n_neighbors_range, weights):
        super().__init__()
        self.cv = cv
        self.n_neighbors_range = n_neighbors_range
        self.weights = weights

    def fit(self, X_train, y_train):
        self._check_fit_args(X_train, y_train)
        self.n_predictors = X_train.shape[1]
        self.fitted = True

        model = KNeighborsRegressor()
        param_grid = {'n_neighbors': self.n_neighbors_range,
                      'weights': self.weights}
        cv_model = GridSearchCV(model, param_grid, cv=self.cv)
        self.model = cv_model.fit(X_train, y_train)

        return self

    def predict(self, X_pred):
        self._check_predict_args(X_pred)

        return self.model.predict(X_pred)


class RandomForestRegressionCV(BlankMLModel):
    """
    Object to produce 1-step ahead forecasts using
    cross-validated random forest regression
    """

    def __init__(self,
                 cv,
                 n_estimators=None,
                 max_features=None,
                 min_samples_leaf=None,
                 bootstrap=None,
                 max_depth=None,
                 cv_method='grid',
                 cv_n_iter=None,
                 random_seed=None):
        super().__init__()
        self.cv = cv
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.random_seed = random_seed

        if self.n_estimators is None:
            self.n_estimators = [100]
        if self.max_features is None:
            self.max_features = [1.0]
        if self.min_samples_leaf is None:
            self.min_samples_leaf = [1]
        if self.bootstrap is None:
            self.bootstrap = [True]
        if self.max_depth is None:
            self.max_depth = [None]

        if self.random_seed is not None:
            self.random_seed = [random_seed]
        else:
            self.random_seed = [None]

        if cv_method not in ['grid', 'random']:
            raise ValueError(''' cv_method must be either 'grid' or 'random' ''')
        if cv_method == 'random' and (cv_n_iter is None or cv_n_iter < 1):
            raise ValueError(''' invalid value of cv_n_iter ''')

        self.cv_method = cv_method
        self.cv_n_iter = cv_n_iter

    def fit(self, X_train, y_train):
        self._check_fit_args(X_train, y_train)
        self.n_predictors = X_train.shape[1]
        self.fitted = True

        model = RandomForestRegressor()
        param_grid = {'n_estimators': self.n_estimators,
                      'max_features': self.max_features,
                      'min_samples_leaf': self.min_samples_leaf,
                      'bootstrap': self.bootstrap,
                      'max_depth': self.max_depth,
                      'random_state': self.random_seed}
        if self.cv_method == 'grid':
            cv_model = GridSearchCV(model,
                                    param_grid,
                                    cv=self.cv)
        else:
            cv_model = RandomizedSearchCV(model,
                                          param_grid,
                                          cv=self.cv,
                                          n_iter=self.cv_n_iter,
                                          random_state=self.random_seed[0])
        self.model = cv_model.fit(X_train, y_train)

        return self

    def predict(self, X_pred):
        self._check_predict_args(X_pred)

        return self.model.predict(X_pred)


class SVRCV(BlankMLModel):
    """
    Object to produce 1-step ahead forecasts using
    cross-validated support vector regression
    """

    def __init__(self,
                 cv,
                 kernel=None,
                 gamma=None,
                 C=None,
                 cv_method='grid',
                 cv_n_iter=None,
                 random_seed=None):
        super().__init__()

        self.cv = cv

        self.kernel = kernel
        self.gamma = gamma
        self.C = C

        self.random_seed = random_seed

        if self.kernel is None:
            self.kernel = ['rbf']
        if self.gamma is None:
            self.gamma = ['scale']
        if self.C is None:
            self.C = [1.0]

        if self.random_seed is not None:
            self.random_seed = [random_seed]
        else:
            self.random_seed = [None]

        if cv_method not in ['grid', 'random']:
            raise ValueError(''' cv_method must be either 'grid' or 'random' ''')
        if cv_method == 'random' and (cv_n_iter is None or cv_n_iter < 1):
            raise ValueError(''' invalid value of cv_n_iter ''')

        self.cv_method = cv_method
        self.cv_n_iter = cv_n_iter

    def fit(self, X_train, y_train):
        self._check_fit_args(X_train, y_train)
        self.n_predictors = X_train.shape[1]
        self.fitted = True

        model = SVR()
        param_grid = {'kernel': self.kernel,
                      'gamma': self.gamma,
                      'C': self.C}
        if self.cv_method == 'grid':
            cv_model = GridSearchCV(model,
                                    param_grid,
                                    cv=self.cv)
        else:
            cv_model = RandomizedSearchCV(model,
                                          param_grid,
                                          cv=self.cv,
                                          n_iter=self.cv_n_iter,
                                          random_state=self.random_seed[0])
        self.model = cv_model.fit(X_train, y_train)

        return self

    def predict(self, X_pred):
        self._check_predict_args(X_pred)

        return self.model.predict(X_pred)


class AdaBoostRegressorCV(BlankMLModel):
    """
    Object to produce 1-step ahead forecasts using
    cross-validated support vector regression
    """

    def __init__(self,
                 cv,
                 n_estimators=None,
                 learning_rate=None,
                 loss=None,
                 max_depth=None,
                 cv_method='grid',
                 cv_n_iter=None,
                 random_seed=None):
        super().__init__()

        self.cv = cv

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.max_depth = max_depth

        self.random_seed = random_seed

        if self.n_estimators is None:
            self.n_estimators = [50]
        if self.learning_rate is None:
            self.learning_rate = [1.0]
        if self.loss is None:
            self.loss = ['linear']
        if self.max_depth is None:
            self.max_depth = [3]

        if self.random_seed is not None:
            self.random_seed = [random_seed]
        else:
            self.random_seed = [None]

        if cv_method not in ['grid', 'random']:
            raise ValueError(''' cv_method must be either 'grid' or 'random' ''')
        if cv_method == 'random' and (cv_n_iter is None or cv_n_iter < 1):
            raise ValueError(''' invalid value of cv_n_iter ''')

        self.cv_method = cv_method
        self.cv_n_iter = cv_n_iter

    def fit(self, X_train, y_train):
        self._check_fit_args(X_train, y_train)
        self.n_predictors = X_train.shape[1]
        self.fitted = True

        model = AdaBoostRegressor()
        param_grid = {'n_estimators': self.n_estimators,
                      'learning_rate': self.learning_rate,
                      'loss': self.loss,
                      'base_estimator': [DecisionTreeRegressor(max_depth=i) for i in self.max_depth]}
        if self.cv_method == 'grid':
            cv_model = GridSearchCV(model,
                                    param_grid,
                                    cv=self.cv)
        else:
            cv_model = RandomizedSearchCV(model,
                                          param_grid,
                                          cv=self.cv,
                                          n_iter=self.cv_n_iter,
                                          random_state=self.random_seed[0])
        self.model = cv_model.fit(X_train, y_train)

        return self

    def predict(self, X_pred):
        self._check_predict_args(X_pred)

        return self.model.predict(X_pred)