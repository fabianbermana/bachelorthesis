# standard imports
import random
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# external imports
import numpy as np
import cvxopt
import matlab
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import TimeSeriesSplit


class BlankModel:
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


class BMA(BlankModel):
    """
    Object to produce 1-step ahead forecasts using
    Bayesian Model Averaging
    """

    def __init__(self):
        super().__init__()

    ### -------------------------- ###

    def _BIC(self, ls_model, X_train, y_train):
        T = X_train.shape[0]
        N = X_train.shape[1]
        fitted_y = ls_model.predict(X_train)
        resid = y_train - fitted_y
        sigma = np.var(resid, ddof=0)

        return T * np.log(sigma) + np.log(T) * N

    ### -------------------------- ###

    def fit(self, X_train, y_train):
        """
        Fit the BMA model

        Parameters
        ----------
        X_train: Numpy array, Predictor to train the model on

        y_train: Numpy vector, Dependent variable to train the model on

        Returns
        -------
        self, the trained model
        """
        self._check_fit_args(X_train, y_train)
        self.n_predictors = X_train.shape[1]
        self.fitted = True

        # compute all combinations
        indices = list(range(0, self.n_predictors))
        self.combinations = []
        for n in range(1, self.n_predictors + 1):
            self.combinations.extend(list(itertools.combinations(indices, n)))

        n_models = len(self.combinations)

        # compute linear models and bic
        ls_models = np.ndarray((n_models,), LinearRegression)
        ls_bic = np.zeros((n_models,))
        for i, combination in enumerate(self.combinations):
            ls_models[i] = LinearRegression(fit_intercept=False).fit(X_train[:, combination], y_train)

        def calc_bic(combinations, X_train, y_train):
            ls_bic = np.zeros((n_models,))
            for i, combination in enumerate(combinations):
                ls_bic[i] = self._BIC(ls_models[i], X_train[:, combination], y_train)
            return ls_bic

        ls_bic = calc_bic(self.combinations, X_train, y_train)

        self.ls_models = ls_models
        self.ls_bic = ls_bic

        def calculate_BMA_weights(ls_bic):
            weights_BMA = np.zeros((n_models,))
            for i in range(n_models):
                bic_i = ls_bic[i]
                denom = np.sum(np.exp(0.5 * (bic_i - ls_bic)))
                weights_BMA[i] = 1 / denom

            return weights_BMA

        self.weights_BMA = calculate_BMA_weights(self.ls_bic)

        return self

    def predict(self, X_pred):
        """
        Produce a forecast based on the trained model

        Parameters
        ----------
        X_pred: Numpy array, 1 x (# of predictors) array with the same predictors used to fit the model

        Returns
        -------
        Numpy array, 1x1 size the point prediction
        """
        self._check_predict_args(X_pred)

        prediction = np.zeros((1,))
        n_models = len(self.ls_models)

        for i, combination in enumerate(self.combinations):
            prediction += self.weights_BMA[i] * self.ls_models[i].predict(X_pred[[0], combination].reshape(1, -1))

        return prediction


class MMA(BlankModel):
    """
    Object to produce 1-step ahead forecasts using
    Mallows Model Averaging. A port of the matlab code
    written by Chu-An Liu and Bruce E. Hansen from
    the University of Wisconsin
    https://www.ssc.wisc.edu/~bhansen/progs/joe_12.html
    """

    def __init__(self):
        super().__init__()

    def fit(self, X_train, y_train):
        """
        Fit the MMA model

        Parameters
        ----------
        X_train: Numpy array, Predictor to train the model on

        y_train: Numpy vector, Dependent variable to train the model on

        Returns
        -------
        self, the trained model
        """
        self._check_fit_args(X_train, y_train)
        self.n_predictors = X_train.shape[1]
        self.fitted = True

        n_train = X_train.shape[0]

        ### VARIABLE SELECTION ###

        # use lasso to see which regressors will be included
        cv = TimeSeriesSplit(n_splits=5)
        lasso = LassoCV(cv=cv).fit(X_train, y_train)
        included = lasso.coef_ > 0

        # sort variables based on lasso beta values
        # assumes input data have been standardized
        lasso_selected = []

        for i in range(len(included)):
            if included[i]:
                lasso_selected.append((i, lasso.coef_[i]))

        lasso_selected = sorted(lasso_selected, key=lambda lst: lst[1], reverse=True)
        lasso_selected = [item[0] for item in lasso_selected]

        # randomly sort zero-coeff regressors
        leftover = set(range(0, self.n_predictors)).difference(set(lasso_selected))
        leftover = list(leftover)
        random.shuffle(leftover)

        # final ordering of nested models
        lasso_selected.extend(leftover)
        self.order = np.array(lasso_selected)
        self.model_regressors = []
        for i in range(self.n_predictors + 1):
            self.model_regressors.append(self.order[:i])

        s = np.zeros((self.n_predictors + 1, self.n_predictors))
        for i in range(self.n_predictors + 1):
            regr = self.model_regressors[i]
            for j in regr:
                s[i, j] = 1

        ### MMA PART ###
        y_train = y_train.reshape(-1, 1)
        n, p = X_train.shape
        m = s.shape[0]
        bbeta = np.zeros((p, m))

        for j in range(m):
            # print(j+1)
            # print('')
            ss = np.ones((n, 1)) @ s[[j], :] > 0

            xs = X_train[ss]
            xs = xs.reshape(n, int(len(xs) / n))
            betas = np.linalg.lstsq(xs.T @ xs, xs.T @ y_train, rcond=None)[0]

            for i, num in enumerate(betas):
                bbeta[i, j] = num

            sj = (s[[j], :] > 0) * 1

            # print(sj)
            # print(sj.shape)
            # print(betas.shape)
            # print(bbeta[sj,j].shape)
            # print('')

            # bbeta[sj,j][:,0:j] = betas.flatten()

            # bbeta[sj.reshape(-1,1)] = betas

        ee = y_train @ np.ones((1, m)) - X_train @ bbeta
        ehat = y_train - X_train @ bbeta[:, [m - 1]]
        sighat = (ehat.T @ ehat) / (n - p)

        a1 = ee.T @ ee
        a2 = (np.sum(s, axis=1) * sighat).T

        w0 = np.ones((m, 1)) / m

        # print(cvxopt.matrix(0, (1,1)).size)

        P = cvxopt.matrix(a1, a1.shape)
        q = cvxopt.matrix(a2, a2.shape)

        G = np.zeros((1, m))
        G = np.vstack([G, np.diag(np.ones((13,)))])
        G = np.vstack([G, -np.diag(np.ones((13,)))])
        G = cvxopt.matrix(G, G.shape, 'd')

        h = np.zeros((1, 1))
        h = np.vstack([h, np.ones((m, 1))])
        h = np.vstack([h, np.zeros((m, 1))])
        h = cvxopt.matrix(h, h.shape, 'd')

        A = cvxopt.matrix(np.ones((1, m)), (1, m))
        b = cvxopt.matrix(1.0, (1, 1))

        cvxopt.solvers.options['show_progress'] = False
        w = cvxopt.solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b, initvals=w0)

        self.mallows_weight = np.array(w['x'])

        self.betahat = bbeta @ self.mallows_weight

        return self

    def predict(self, X_pred):
        """
        Produce a forecast based on the trained model

        Parameters
        ----------
        X_pred: Numpy array, 1 x (# of predictors) array with the same predictors used to fit the model

        Returns
        -------
        Numpy array, 1x1 size the point prediction
        """
        self._check_predict_args(X_pred)

        return X_pred @ self.betahat


class JMA(BlankModel):
    """
    Object to produce 1-step ahead forecasts using
    Jackknife Model Averaging. A port of the matlab code
    written by Chu-An Liu and Bruce E. Hansen from
    the University of Wisconsin
    https://www.ssc.wisc.edu/~bhansen/progs/joe_12.html
    """

    def __init__(self):
        super().__init__()

    def fit(self, X_train, y_train):
        """
        Fit the JMA model

        Parameters
        ----------
        X_train: Numpy array, Predictor to train the model on

        y_train: Numpy vector, Dependent variable to train the model on

        Returns
        -------
        self, the trained model
        """
        self._check_fit_args(X_train, y_train)
        self.n_predictors = X_train.shape[1]
        self.fitted = True

        n_train = X_train.shape[0]

        ### VARIABLE SELECTION ###

        # use lasso to see which regressors will be included
        cv = TimeSeriesSplit(n_splits=5)
        lasso = LassoCV(cv=cv).fit(X_train, y_train)
        included = lasso.coef_ > 0

        # sort variables based on lasso beta values
        # assumes input data have been standardized
        lasso_selected = []

        for i in range(len(included)):
            if included[i]:
                lasso_selected.append((i, lasso.coef_[i]))

        lasso_selected = sorted(lasso_selected, key=lambda lst: lst[1], reverse=True)
        lasso_selected = [item[0] for item in lasso_selected]

        # randomly sort zero-coeff regressors
        leftover = set(range(0, self.n_predictors)).difference(set(lasso_selected))
        leftover = list(leftover)
        random.shuffle(leftover)

        # final ordering of nested models
        lasso_selected.extend(leftover)
        self.order = np.array(lasso_selected)
        self.model_regressors = []
        for i in range(self.n_predictors + 1):
            self.model_regressors.append(self.order[:i])

        s = np.zeros((self.n_predictors + 1, self.n_predictors))
        for i in range(self.n_predictors + 1):
            regr = self.model_regressors[i]
            for j in regr:
                s[i, j] = 1

        ### JMA PART ###
        y_train = y_train.reshape(-1, 1)
        n, p = X_train.shape
        m = s.shape[0]
        bbeta = np.zeros((p, m))

        ee = np.zeros((n, m))

        for j in range(m):
            ss = np.ones((n, 1)) @ s[[j], :] > 0
            xs = X_train[ss]
            xs = xs.reshape(n, int(len(xs) / n))
            betas = np.linalg.lstsq(xs.T @ xs, xs.T @ y_train, rcond=None)[0]

            for i, num in enumerate(betas):
                bbeta[i, j] = num

            sj = (s[[j], :] > 0) * 1

            ei = y_train - xs @ betas
            temp = np.linalg.lstsq(xs.T, (xs.T @ xs).T, rcond=None)[0] @ xs.T
            hi = np.diag(temp).reshape(-1, 1)
            ee[:, [j]] = ei * (1 / (1 - hi))

        a1 = ee.T @ ee
        a2 = np.zeros((m, 1))

        w0 = np.ones((m, 1)) / m

        P = cvxopt.matrix(a1, a1.shape)
        q = cvxopt.matrix(a2, a2.shape)

        G = np.zeros((1, m))
        G = np.vstack([G, np.diag(np.ones((13,)))])
        G = np.vstack([G, -np.diag(np.ones((13,)))])
        G = cvxopt.matrix(G, G.shape, 'd')

        h = np.zeros((1, 1))
        h = np.vstack([h, np.ones((m, 1))])
        h = np.vstack([h, np.zeros((m, 1))])
        h = cvxopt.matrix(h, h.shape, 'd')

        A = cvxopt.matrix(np.ones((1, m)), (1, m))
        b = cvxopt.matrix(1.0, (1, 1))

        cvxopt.solvers.options['show_progress'] = False
        w = cvxopt.solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b, initvals=w0)

        self.mallows_weight = np.array(w['x'])

        self.betahat = bbeta @ self.mallows_weight

        return self

    def predict(self, X_pred):
        """
        Produce a forecast based on the trained model

        Parameters
        ----------
        X_pred: Numpy array, 1 x (# of predictors) array with the same predictors used to fit the model

        Returns
        -------
        Numpy array, 1x1 size the point prediction
        """
        self._check_predict_args(X_pred)

        return X_pred @ self.betahat


class WALS(BlankModel):
    """
    Object to produce 1-step ahead forecasts using
    Weighted Average Least Squares. Acts as a wrapper
    to call the matlab function of Magnus et al
    (2010, 2016), downloaded from
    https://www.janmagnus.nl/items/WALS.pdf
    """

    def __init__(self, matlab_engine):
        super().__init__()
        self.matlab_engine = matlab_engine

    def fit(self, X_train, y_train):
        """
        Fit the WALS model

        Parameters
        ----------
        X_train: Numpy array, Predictor to train the model on

        y_train: Numpy vector, Dependent variable to train the model on

        Returns
        -------
        self, the trained model
        """
        self._check_fit_args(X_train, y_train)
        self.n_predictors = X_train.shape[1]
        self.fitted = True

        y = matlab.double(y_train.reshape(-1, 1).tolist())
        X1 = matlab.double(np.ones((X_train.shape[0], 1)).tolist())
        X2 = matlab.double(X_train.tolist())

        matlab_output = self.matlab_engine.wals(y, X1, X2)

        self.beta = np.array(matlab_output)

        return self

    def predict(self, X_pred):
        """
        Produce a forecast based on the trained model

        Parameters
        ----------
        X_pred: Numpy array, 1 x (# of predictors) array with the same predictors used to fit the model

        Returns
        -------
        Numpy array, 1x1 size the point prediction
        """
        self._check_predict_args(X_pred)
        X_pred = np.hstack([np.ones((1, 1)), X_pred])
        return X_pred @ self.beta