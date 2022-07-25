import torch
from tqdm.auto import tqdm
from IPython.display import clear_output
# Модули
import config as conf
import ploting as pl
import log_history as lg
import results as res

def train(model, optimizer, train_dataloader, val_dataloader, criterion=conf.cr, n_epochs=conf.def_n_ep, device=conf.dev):
    '''INPUT
            -> model: model
            -> criterion: nn.CrossEntropyLoss
            -> optimizer: optim.SGD(model.parameters(), momentum=0.95, lr=0.1)
            -> train_dataloader
            -> val_dataloader
            -> n_epochs: number of epochs
            -> history_dict: empty dict
            -> device: torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            -> scheduler: torch.optim.lr_scheduler.CosineAnnealingLR
        OUTPUT
            -> trainded model
    '''
    best_score = [0, 0]  # [num_epoch, best_score]
    train_loss_log, train_acc_log, val_loss_log, val_acc_log = [], [], [], []
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0.0001, last_epoch=-1,
                                                           verbose=False)

    for epoch in range(n_epochs):
        # тренировка
        train_epoch_loss, train_epoch_true_hits = torch.empty(0), torch.empty(0)
        model.train()
        for imgs, labels_ in tqdm(train_dataloader, desc=f"Training, epoch {epoch}", leave=False):
            imgs, labels = imgs.to(device), labels_.to(device)

            y_pred = model(imgs)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # log loss for the current epoch and the whole training history
            train_epoch_loss = torch.cat((train_epoch_loss.cpu(), loss.cpu().unsqueeze(0) / labels.cpu().size(0)))
            train_loss_log.append(loss.data.cpu() / labels_.size(0))

            # log accuracy for the current epoch and the whole training history
            pred_classes = torch.argmax(y_pred, dim=-1).cpu()
            train_epoch_true_hits = torch.cat((
                train_epoch_true_hits,
                (pred_classes.cpu() == labels_).sum().unsqueeze(0)
            ))
            train_acc_log.append((pred_classes == labels_).sum() / labels_.shape[0])

        # валидация
        val_epoch_loss, val_epoch_true_hits = torch.empty(0), torch.empty(0)
        model.eval()
        with torch.no_grad():
            for imgs, labels_ in tqdm(val_dataloader, desc=f"Validating, epoch {epoch}", leave=False):
                imgs, labels = imgs.to(device), labels_.to(device)

                y_pred = model(imgs)
                loss = criterion(y_pred, labels)
                val_epoch_loss = torch.cat((val_epoch_loss.cpu(), loss.cpu().unsqueeze(0) / labels.cpu().size(0)))

                pred_classes = torch.argmax(y_pred, dim=-1).cpu()
                val_epoch_true_hits = torch.cat((
                    val_epoch_true_hits,
                    (pred_classes == labels_).sum().unsqueeze(0)
                ))

        val_loss_log.append(val_epoch_loss.cpu().mean())
        val_acc_log.append(
            val_epoch_true_hits.cpu().sum() / val_epoch_true_hits.cpu().size(0) / val_dataloader.batch_size)
        clear_output()
        pl.plot_history(train_loss_log, val_loss_log, "loss")
        pl.plot_history(train_acc_log, val_acc_log, "accuracy")

        TrLs = train_epoch_loss.mean().item()
        TrAc = (train_epoch_true_hits.sum() / train_epoch_true_hits.size(0) / train_dataloader.batch_size).item()
        VaLs = val_epoch_loss.mean().item()
        VaAc = (val_epoch_true_hits.sum() / val_epoch_true_hits.size(0) / val_dataloader.batch_size).item()

        lg.log_history(epoch, train_loss_log, val_loss_log, train_acc_log, val_acc_log, TrLs, TrAc, VaLs, VaAc)

        print("Train loss:", TrLs)
        print("Train acc:", TrAc)
        print("Val loss:", VaLs)
        print("Val acc:", VaAc)

        # будем сохранять лучший результат
        if VaAc >= best_score[1]:
            best_score[0], best_score[1] = epoch, VaAc

    return best_score