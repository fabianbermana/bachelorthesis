# standard imports
import time

# external imports
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# own imports
from forecast_evaluation import R2OS, CW_test, MSE, MAE, R2LOG


def expanding_window_parallel(X_train, X_test, y_train, y_test, forecast_model, model_args=None, workers=-1,
                              verbosity=10, random_seeds=None):
    """
    DESCRIPTION
    -----------
    Produce 1-step ahead forecasts, trained with recursive expanding windows.
    Uses parallelization on multiple CPU cores to speed up the process.

    PARAMETERS
    -----------
    X_train: 2-dimensional numpy array, independent variables used to train the forecast model

    X_test: 2-dimensional numpy array, independent variables used for forecasting

    y_train: 1-dimensional numpy array, dependent variable used to train the forecast model

    y_test: 1-dimensional numpy array, dependent variable to forecast

    forecast_model: python object, model used for forecasting. Make this object similar to the

    sklearn models, with a fit() and predict() function

    model_args: dict, arguments to pass when initializing the model

    workers: int, how many processes are used. -1 denotes using all available logical processors
    """

    start_time = time.time()

    # if no model arguments, parse an empty dictionary
    if model_args is None:
        model_args = {}

    n_periods = y_test.shape[0]
    pred_model = np.zeros((n_periods,))
    pred_bmk = np.zeros((n_periods,))

    if random_seeds is not None:
        if len(random_seeds) != n_periods:
            raise ValueError('length of random_seeds must be same as length of X_test')

    # function to produce recursive expanding window forecasts
    def ExpW_forecast(X_train, y_train, i, random_seed=None):
        # print(f'{i + 1}/{n_periods}', end='\r')

        if random_seed is not None:
            model_args['random_seed'] = random_seed

        # take the correct subsets of the dataset
        X_train = np.vstack([X_train, X_test[:i, :]])
        y_train = np.hstack([y_train, y_test[:i]])

        # scale the data
        X_scaler = StandardScaler().fit(X_train)
        X = X_scaler.transform(X_train)
        y = y_train.reshape(-1, 1)
        y_scaler = StandardScaler().fit(y)
        y = y_scaler.transform(y).flatten()

        # forecasts of model
        model = forecast_model(**model_args).fit(X, y)
        pred = model.predict(X_scaler.transform(X_test[[i], :]))

        # historical average forecast as the baseline
        bmk = np.mean(y_train)

        # return a tuple of the baseline and the expanding window forecasts
        return bmk, y_scaler.inverse_transform(pred.reshape(1,-1))[0]

    # run the forecast models in parallel
    if random_seeds is None:
        pred_pairs = Parallel(n_jobs=workers, backend='loky', verbose=verbosity) \
            (delayed(ExpW_forecast)(X_train, y_train, i) for i in range(n_periods))
    else:
        pred_pairs = Parallel(n_jobs=workers, backend='loky', verbose=verbosity) \
            (delayed(ExpW_forecast)(X_train, y_train, i, random_seeds[i]) for i in range(n_periods))

    # split the prediction tuples
    for i, pred_pair in enumerate(pred_pairs):
        pred_bmk[i] = pred_pair[0]
        pred_model[i] = pred_pair[1]

    # calculate metrics
    MSPE_model = metrics.mean_squared_error(y_test, pred_model)
    MSPE_bmk = metrics.mean_squared_error(y_test, pred_bmk)
    R2OS_val = R2OS(MSPE_model, MSPE_bmk)
    cw_test= CW_test(y_test, pred_bmk, pred_model)
    losses = {
        'MSE': MSE(y_test, pred_model),
        'MAE': MAE(y_test, pred_model),
        'R2LOG': R2LOG(y_test, pred_model)
    }

    end_time = time.time()
    elapsed_time = end_time - start_time

    return {'out-of-sample R2': R2OS_val,
            'MSPE_bmk': MSPE_bmk,
            'MSPE_model': MSPE_model,
            'losses': losses,
            'pred_bmk': pred_bmk,
            'pred_model': pred_model,
            'CW_test': cw_test,
            'elapsed time': elapsed_time}


def AveW_parallel(X_train, X_test, y_train, y_test, forecast_model, m, w_min, model_args=None, workers=-1,
                  verbosity=10, random_seeds=None):
    """
    DESCRIPTION
    -----------
    Produce 1-step ahead forecasts, trained with recursive expanding windows.
    Uses parallelization on multiple CPU cores to speed up the process.

    PARAMETERS
    -----------
    X_train: 2-dimensional numpy array, independent variables used to train the forecast model

    X_test: 2-dimensional numpy array, independent variables used for forecasting

    y_train: 1-dimensional numpy array, dependent variable used to train the forecast model

    y_test: 1-dimensional numpy array, dependent variable to forecast

    forecast_model: python object, model used for forecasting. Make this object similar to the

    m: number of windows for AveW forecasting

    w_min: minimum AveW window size

    sklearn models, with a fit() and predict() function

    model_args: dict, arguments to pass when initializing the model

    workers: int, how many processes are used. -1 denotes using all available logical processors
    """

    start_time = time.time()

    # parse an empty dict as the additional arguments
    if model_args is None:
        model_args = {}

    n_periods = y_test.shape[0]

    if random_seeds is not None:
        if len(random_seeds) != n_periods:
            raise ValueError('length of random_seeds must be same as length of X_test')

    # arrays to store forecasts
    pred_model = np.zeros((n_periods,))
    pred_bmk = np.zeros((n_periods,))

    # loop through all forecasting periods
    def AveW_forecast(X_train, y_train, j, m, random_seed=None):
        print(f'{j + 1}/{n_periods}', end='\r')

        if random_seed is not None:
            model_args['random_seed'] = random_seed

        # properly load the subsets for each point forecast
        X_train = np.vstack([X_train, X_test[:j, :]])
        y_train = np.hstack([y_train, y_test[:j]])

        T = X_train.shape[0]
        single_preds = np.zeros((m,))

        # AveW part
        for i in range(1, m + 1):
            # calculate size of window i
            w_i = int(w_min + (i - 1) / (m - 1) * (T - w_min)) - 1
            # print(w_i)
            # starting and ending index of window i
            start, end = T - 1 - w_i, T
            # print(f'start:{start}, end:{end}')

            # scale the data
            X = X_train[start:end, :]
            X_scaler = StandardScaler().fit(X)
            X = X_scaler.transform(X)
            y = y_train[start:end].reshape(-1, 1)
            y_scaler = StandardScaler().fit(y)
            y = y_scaler.transform(y).flatten()

            # train model and predict
            model = forecast_model(**model_args).fit(X, y)
            pred = model.predict(X_scaler.transform(X_test[[j], :]))
            single_preds[i - 1] = y_scaler.inverse_transform(pred.reshape(1,-1))

        # historical average is the benchmark
        bmk = np.mean(y_train)

        # return tuple of hist. avg and AveW forecast
        return bmk, np.mean(single_preds)

    # run predictions with multiple processors
    if random_seeds is None:
        pred_pairs = Parallel(n_jobs=workers, backend='loky', verbose=verbosity) \
            (delayed(AveW_forecast)(X_train, y_train, j, m) for j in range(n_periods))
    else:
        pred_pairs = Parallel(n_jobs=workers, backend='loky', verbose=verbosity) \
            (delayed(AveW_forecast)(X_train, y_train, j, m, random_seeds[j]) for j in range(n_periods))

    # split tuple of predictions
    for i, pred_pair in enumerate(pred_pairs):
        pred_bmk[i] = pred_pair[0]
        pred_model[i] = pred_pair[1]

    # calculate metrics
    MSPE_model = metrics.mean_squared_error(y_test, pred_model)
    MSPE_bmk = metrics.mean_squared_error(y_test, pred_bmk)
    R2OS_val = R2OS(MSPE_model, MSPE_bmk)
    cw_test= CW_test(y_test, pred_bmk, pred_model)
    losses = {
        'MSE': MSE(y_test, pred_model),
        'MAE': MAE(y_test, pred_model),
        'R2LOG': R2LOG(y_test, pred_model)
    }

    end_time = time.time()
    elapsed_time = end_time - start_time

    return {'out-of-sample R2': R2OS_val,
            'MSPE_bmk': MSPE_bmk,
            'MSPE_model': MSPE_model,
            'losses': losses,
            'pred_bmk': pred_bmk,
            'pred_model': pred_model,
            'CW_test': cw_test,
            'elapsed time': elapsed_time}


def HA_combination(y_train, y_test, y_pred, delta=0.5):
    """
    DESCRIPTION
    -----------
    Evaluate the forecast combination method of Zhang et al (2020) where
    forecasts of a historical model are combined with the historical average

    PARAMETERS
    -----------
    y_train: 1-dimensional numpy array, dependent variable used to train the forecast model

    y_test: 1-dimensional numpy array, dependent variable to forecast

    y_pred: 1-dimensional numpy array, predictions of a sophisticated model

    delta: float, value in range [0,1], forecast combination weight
    """

    start_time = time.time()
    n_periods = y_test.shape[0]
    pred_model = np.zeros((n_periods,))
    pred_bmk = np.zeros((n_periods,))

    # run a forecast for every test period
    for i in range(n_periods):
        print(f'{i + 1}/{n_periods}', end='\r')

        # forecasts of historical average
        bmk = np.mean(y_train)
        pred_bmk[i] = bmk

        # add new row to training data
        y_train = np.hstack([y_train, y_test[i]])

        # combination forecast
        pred_model[i] = (1 - delta) * bmk + delta * y_pred[i]


    # calculate metrics
    MSPE_model = metrics.mean_squared_error(y_test, pred_model)
    MSPE_bmk = metrics.mean_squared_error(y_test, pred_bmk)
    R2OS_val = R2OS(MSPE_model, MSPE_bmk)
    cw_test= CW_test(y_test, pred_bmk, pred_model)
    losses = {
        'MSE': MSE(y_test, pred_model),
        'MAE': MAE(y_test, pred_model),
        'R2LOG': R2LOG(y_test, pred_model)
    }

    end_time = time.time()
    elapsed_time = end_time - start_time

    return {'out-of-sample R2': R2OS_val,
            'MSPE_bmk': MSPE_bmk,
            'MSPE_model': MSPE_model,
            'losses': losses,
            'pred_bmk': pred_bmk,
            'pred_model': pred_model,
            'CW_test': cw_test,
            'elapsed time': elapsed_time}


def expanding_window(X_train, X_test, y_train, y_test, forecast_model, model_args=None, random_seeds=None):
    """
    parameters
    -----------
    X_train: 2-dimensional numpy array, independent variables used to train the forecast model
    X_test: 2-dimensional numpy array, independent variables used for forecasting
    y_train: 1-dimensional numpy array, dependent variable used to train the forecast model
    y_test: 1-dimensional numpy array, dependent variable to forecast
    forecast_model: python object, model used for forecasting. Make this object similar to the
    sklearn models, with a fit() and predict() function
    model_args: dict, arguments to pass when initializing the model
    """
    start_time = time.time()
    if model_args is None:
        model_args = {}
    n_periods = y_test.shape[0]
    pred_model = np.zeros((n_periods,))
    pred_bmk = np.zeros((n_periods,))

    if random_seeds is not None:
        if len(random_seeds) != n_periods:
            raise ValueError('length of random_seeds must be same as length of X_test')

    # run a forecast for every test period
    for i in range(n_periods):
        print(f'{i + 1}/{n_periods}', end='\r')

        if random_seeds is not None:
            model_args['random_seed'] = random_seeds[i]

        X_scaler = StandardScaler().fit(X_train)
        X = X_scaler.transform(X_train)
        y = y_train.reshape(-1, 1)
        y_scaler = StandardScaler().fit(y)
        y = y_scaler.transform(y).flatten()

        # forecasts of model
        model = forecast_model(**model_args).fit(X, y)
        pred = model.predict(X_scaler.transform(X_test[[i], :]))
        pred_model[i] = y_scaler.inverse_transform(pred.reshape(1,-1))

        # forecasts of historical average
        bmk = np.mean(y_train)
        pred_bmk[i] = bmk

        # add new row to training data
        X_train = np.vstack([X_train, X_test[[i], :]])
        y_train = np.hstack([y_train, y_test[i]])


    # calculate metrics
    MSPE_model = metrics.mean_squared_error(y_test, pred_model)
    MSPE_bmk = metrics.mean_squared_error(y_test, pred_bmk)
    R2OS_val = R2OS(MSPE_model, MSPE_bmk)
    cw_test= CW_test(y_test, pred_bmk, pred_model)
    losses = {
        'MSE': MSE(y_test, pred_model),
        'MAE': MAE(y_test, pred_model),
        'R2LOG': R2LOG(y_test, pred_model)
    }

    end_time = time.time()
    elapsed_time = end_time - start_time

    return {'out-of-sample R2': R2OS_val,
            'MSPE_bmk': MSPE_bmk,
            'MSPE_model': MSPE_model,
            'losses': losses,
            'pred_bmk': pred_bmk,
            'pred_model': pred_model,
            'CW_test': cw_test,
            'elapsed time': elapsed_time}


def AveW(X_train, X_test, y_train, y_test, forecast_model, m, w_min, model_args=None, random_seeds=None):
    start_time = time.time()

    if model_args is None:
        model_args = {}

    n_periods = y_test.shape[0]
    pred_model = np.zeros((n_periods,))
    pred_bmk = np.zeros((n_periods,))

    if random_seeds is not None:
        if len(random_seeds) != n_periods:
            raise ValueError('length of random_seeds must be same as length of X_test')

    # loop through all forecasting periods
    for j in range(n_periods):
        print(f'{j + 1}/{n_periods}', end='\r')

        if random_seeds is not None:
            model_args['random_seed'] = random_seeds[j]

        T = X_train.shape[0]
        single_preds = np.zeros((m,))

        # AveW part
        for i in range(1, m + 1):
            # calculate size of window i
            w_i = int(w_min + (i - 1) / (m - 1) * (T - w_min)) - 1
            # print(w_i)
            # starting and ending index of window i
            start, end = T - 1 - w_i, T
            #print(f'start:{start}, end:{end}')

            X = X_train[start:end, :]
            X_scaler = StandardScaler().fit(X)
            X = X_scaler.transform(X)
            y = y_train[start:end].reshape(-1, 1)
            y_scaler = StandardScaler().fit(y)
            y = y_scaler.transform(y).flatten()

            model = forecast_model(**model_args).fit(X, y)
            pred = model.predict(X_scaler.transform(X_test[[j], :]))
            single_preds[i - 1] = y_scaler.inverse_transform(pred.reshape(1,-1))

        pred_model[j] = np.mean(single_preds)

        # forecasts of historical average
        bmk = np.mean(y_train)
        pred_bmk[j] = bmk

        # add new row to training data
        X_train = np.vstack([X_train, X_test[[j], :]])
        y_train = np.hstack([y_train, y_test[j]])


    # calculate metrics
    MSPE_model = metrics.mean_squared_error(y_test, pred_model)
    MSPE_bmk = metrics.mean_squared_error(y_test, pred_bmk)
    R2OS_val = R2OS(MSPE_model, MSPE_bmk)
    cw_test= CW_test(y_test, pred_bmk, pred_model)
    losses = {
        'MSE': MSE(y_test, pred_model),
        'MAE': MAE(y_test, pred_model),
        'R2LOG': R2LOG(y_test, pred_model)
    }

    end_time = time.time()
    elapsed_time = end_time - start_time

    return {'out-of-sample R2': R2OS_val,
            'MSPE_bmk': MSPE_bmk,
            'MSPE_model': MSPE_model,
            'losses': losses,
            'pred_bmk': pred_bmk,
            'pred_model': pred_model,
            'CW_test': cw_test,
            'elapsed time': elapsed_time}


def expanding_window_recurrent(X, y, forecast_model, test_start,
                               model_args=None,
                               random_seeds=None,
                               window_size=12):
    """
    parameters
    -----------
    X_train: 2-dimensional numpy array, independent variables used to train the forecast model
    X_test: 2-dimensional numpy array, independent variables used for forecasting
    y_train: 1-dimensional numpy array, dependent variable used to train the forecast model
    y_test: 1-dimensional numpy array, dependent variable to forecast
    forecast_model: python object, model used for forecasting. Make this object similar to the
    sklearn models, with a fit() and predict() function
    model_args: dict, arguments to pass when initializing the model
    """
    start_time = time.time()
    y = y.reshape(-1,1)
    if model_args is None:
        model_args = {}
    n_periods = y.shape[0] - test_start
    pred_model = np.zeros((n_periods,))
    pred_bmk = np.zeros((n_periods,))


    def tssplit(data, window_size=window_size):
        output = np.zeros((data.shape[0] - window_size, window_size, data.shape[1]))
        for i in range(data.shape[0] - window_size):
            output[i, :, :] = data[i:i + window_size, :]
        return output


    if random_seeds is not None:
        if len(random_seeds) != n_periods:
            raise ValueError('length of random_seeds must be same as length of X_test')

    # run a forecast for every test period
    for i in range(n_periods):
        if ((n_periods+1) % 100 == 0):
            print('')
        print('.', end='')

        if random_seeds is not None:
            model_args['random_seed'] = random_seeds[i]

        X_train = X[0:test_start+i,:]
        y_train = y[0:test_start+i,:]

        X_scaler = StandardScaler().fit(X_train)
        X_current = X_scaler.transform(X_train)
        y_scaler = StandardScaler().fit(y_train)
        y_current = y_scaler.transform(y_train)

        X_temp = X_scaler.transform(X[[test_start+i],:])
        y_temp = y_scaler.transform(y[[test_start+i],:])
        X_current = np.vstack([X_current, X_temp])
        y_current = np.vstack([y_current, y_temp])

        X_rec = tssplit(X_current)
        y_rec = tssplit(y_current)

        # forecasts of model
        model = forecast_model(**model_args).fit(X_rec[0:test_start+i,:,:], y_rec[0:test_start+i,:,:])
        pred = model.predict(X_rec[[-1],:,:])
        pred_model[i] = y_scaler.inverse_transform(pred.reshape(1,-1))

        # forecasts of historical average
        bmk = np.mean(y_train)
        pred_bmk[i] = bmk


    # calculate metrics
    MSPE_model = metrics.mean_squared_error(y[test_start:], pred_model)
    MSPE_bmk = metrics.mean_squared_error(y[test_start:], pred_bmk)
    R2OS_val = R2OS(MSPE_model, MSPE_bmk)
    cw_test= CW_test(y[test_start:], pred_bmk, pred_model)

    end_time = time.time()
    elapsed_time = end_time - start_time

    return {'out-of-sample R2': R2OS_val,
            'MSPE_bmk': MSPE_bmk,
            'MSPE_model': MSPE_model,
            'pred_bmk': pred_bmk,
            'pred_model': pred_model,
            'CW_test': cw_test,
            'elapsed time': elapsed_time}


def expanding_window_recurrent_parallel(X, y, forecast_model, test_start,
                                        model_args=None,
                                        random_seeds=None,
                                        window_size=12,
                                        workers=-1,
                                        verbosity=10):
    """
    parameters
    -----------
    X_train: 2-dimensional numpy array, independent variables used to train the forecast model
    X_test: 2-dimensional numpy array, independent variables used for forecasting
    y_train: 1-dimensional numpy array, dependent variable used to train the forecast model
    y_test: 1-dimensional numpy array, dependent variable to forecast
    forecast_model: python object, model used for forecasting. Make this object similar to the
    sklearn models, with a fit() and predict() function
    model_args: dict, arguments to pass when initializing the model
    """
    start_time = time.time()
    y = y.reshape(-1,1)
    if model_args is None:
        model_args = {}
    n_periods = y.shape[0] - test_start
    pred_model = np.zeros((n_periods,))
    pred_bmk = np.zeros((n_periods,))


    def tssplit(data, window_size=window_size):
        output = np.zeros((data.shape[0] - window_size, window_size, data.shape[1]))
        for i in range(data.shape[0] - window_size):
            output[i, :, :] = data[i:i + window_size, :]
        return output


    if random_seeds is not None:
        if len(random_seeds) != n_periods:
            raise ValueError('length of random_seeds must be same as length of X_test')

    # run a forecast for every test period
    def forecast(X, y, i):
        if ((n_periods+1) % 100 == 0):
            print('')
        print('.', end='')

        if random_seeds is not None:
            model_args['random_seed'] = random_seeds[i]

        X_train = X[0:test_start+i,:]
        y_train = y[0:test_start+i,:]

        X_scaler = StandardScaler().fit(X_train)
        X_current = X_scaler.transform(X_train)
        y_scaler = StandardScaler().fit(y_train)
        y_current = y_scaler.transform(y_train)

        X_temp = X_scaler.transform(X[[test_start+i],:])
        y_temp = y_scaler.transform(y[[test_start+i],:])
        X_current = np.vstack([X_current, X_temp])
        y_current = np.vstack([y_current, y_temp])

        X_rec = tssplit(X_current)
        y_rec = tssplit(y_current)

        # forecasts of model
        model = forecast_model(**model_args).fit(X_rec[0:test_start+i,:,:], y_rec[0:test_start+i,:,:])
        pred = model.predict(X_rec[[-1],:,:])
        pred = y_scaler.inverse_transform(pred.reshape(1,-1))

        # forecasts of historical average
        bmk = np.mean(y_train)
        #pred_bmk[i] = bmk

        return bmk, pred

    # run the forecast models in parallel
    if random_seeds is None:
        pred_pairs = Parallel(n_jobs=workers, backend='loky', verbose=verbosity) \
            (delayed(forecast)(X, y, i) for i in range(n_periods))
    else:
        pred_pairs = Parallel(n_jobs=workers, backend='loky', verbose=verbosity) \
            (delayed(forecast)(X, y, i, random_seeds[i]) for i in range(n_periods))

    # split the prediction tuples
    for i, pred_pair in enumerate(pred_pairs):
        pred_bmk[i] = pred_pair[0]
        pred_model[i] = pred_pair[1]

    # calculate metrics
    MSPE_model = metrics.mean_squared_error(y[test_start:], pred_model)
    MSPE_bmk = metrics.mean_squared_error(y[test_start:], pred_bmk)
    R2OS_val = R2OS(MSPE_model, MSPE_bmk)
    cw_test= CW_test(y[test_start:], pred_bmk, pred_model)

    end_time = time.time()
    elapsed_time = end_time - start_time

    return {'out-of-sample R2': R2OS_val,
            'MSPE_bmk': MSPE_bmk,
            'MSPE_model': MSPE_model,
            'pred_bmk': pred_bmk,
            'pred_model': pred_model,
            'CW_test': cw_test,
            'elapsed time': elapsed_time}