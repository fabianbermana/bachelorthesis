# external imports
import numpy as np


def R2OS(MSFE_M, MSFE_bmk):
    return 1 - (MSFE_M / MSFE_bmk)


def CW_test(y_test, pred_bmk, pred_model):
    p = y_test.shape[0]
    e_1_hat_2 = (y_test - pred_bmk) ** 2
    e_2_hat_2 = (y_test - pred_model) ** 2
    adj = (pred_bmk - pred_model) ** 2

    f_hat = e_1_hat_2 - (e_2_hat_2 - adj)
    f_bar = np.sum(f_hat) / p

    return (p ** 0.5) * f_bar / (np.var(f_hat - f_bar, ddof=1) ** 0.5)


def MSE(y_test, pred_model):
    out = y_test - pred_model
    out = out ** 2
    out = out.mean()
    return out


def MAE(y_test, pred_model):
    out = y_test * pred_model
    out = np.abs(out)
    out = out.mean()
    return out


def R2LOG(y_test, pred_model):
    out = (y_test ** 2) * (pred_model ** -2)
    out = np.log(out)
    out = out.mean()
    return out


def DM_test(y_test, pred_bmk, pred_model):
    p = y_test.shape[0]

    e_1_hat_2 = (y_test - pred_bmk) ** 2
    e_2_hat_2 = (y_test - pred_model) ** 2

    f_hat = e_1_hat_2 - e_2_hat_2
    f_bar = np.sum(f_hat) / p

    return (p ** 0.5) * f_bar / (np.var(f_hat - f_bar, ddof=1) ** 0.5)