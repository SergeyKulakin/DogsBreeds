import results as res

def log_history(num_epoch, train_loss_log, val_loss_log, train_acc_log, val_acc_log, TrLs, TrAc, VaLs, VaAc):
    '''INPUT
            -> history_dict: словарь для записи логов
            -> num_epoch: номер эпохи
            -> train_history: история ибучения batch train
            -> val_history: история ибучения batch val
            -> TrLs: Train loss
            -> TrAc: Train accuracy
            -> VaLs: Val loss
            -> VaAc: Val accuracy
        OUTPUT
            -> dict with model scores and results'''

    res.history_dict[num_epoch] = {'train_loss_log' : train_loss_log,
                               'val_loss_log' : val_loss_log,
                               'train_acc_log' : train_acc_log,
                               'val_acc_log' : val_acc_log,
                               'TrLs' : TrLs,
                               'TrAc' : TrAc,
                               'VaLs' : VaLs,
                               'VaAc' : VaAc}
    return