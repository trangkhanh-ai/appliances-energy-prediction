"""LSTM nhỏ: 18 số đo quá khứ -> 6 số đo tương lai; chọn epoch trong TRAIN."""
import os
import time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import appliances_core as v1

WINDOW = 18
HIDDEN = 24


def inputs(df):
    values = df.Appliances.to_numpy(dtype=np.float32)
    sequence = np.full((len(values), WINDOW, 1), np.nan, dtype=np.float32)
    if len(values) >= WINDOW:
        sequence[WINDOW-1:, :, 0] = sliding_window_view(values, WINDOW)
    calendar = v1.build_features(df)[['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']].to_numpy(np.float32)
    return sequence, calendar


def network():
    import torch
    class TinyLSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(1, HIDDEN, batch_first=True)
            self.output = torch.nn.Linear(HIDDEN + 5, 6)

        def forward(self, sequence, calendar):
            encoded, _ = self.lstm(sequence)
            return self.output(torch.cat((encoded[:, -1, :], calendar), dim=1))
    return TinyLSTM()


def seed_torch(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fit_lstm(df, train_mask, train_end, max_epochs=30, seed=42):
    import torch
    seed_torch(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sequence, calendar = inputs(df)
    labels = v1.build_labels(df).astype(np.float32)
    time_index = pd.DatetimeIndex(df.t)
    cut = pd.Timestamp(train_end) - pd.Timedelta(days=7)
    inner_train = train_mask & (time_index + pd.Timedelta(minutes=60) < cut)
    inner_val = train_mask & (time_index >= cut)
    if inner_train.sum() < 100 or inner_val.sum() < 20:
        raise ValueError('Không đủ dữ liệu TRAIN cho early stopping theo thời gian')
    started = time.perf_counter()

    def train(train, valid=None, epochs=30):
        seed_torch(seed)
        xm, xs = float(sequence[train].mean()), max(float(sequence[train].std()), 1.)
        ym, ys = float(labels[train].mean()), max(float(labels[train].std()), 1.)
        x = torch.tensor((sequence[train]-xm)/xs, device=device)
        cal = torch.tensor(calendar[train], device=device)
        y = torch.tensor((labels[train]-ym)/ys, device=device)
        model = network().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        loss_function = torch.nn.L1Loss()
        history, best, best_epoch, patience = [], float('inf'), epochs, 0
        if valid is not None:
            vx = torch.tensor((sequence[valid]-xm)/xs, device=device)
            vc = torch.tensor(calendar[valid], device=device)
            vy = torch.tensor((labels[valid]-ym)/ys, device=device)
        for epoch in range(epochs):
            model.train()
            order = torch.randperm(len(y), device=device)
            total_loss = 0.
            for indices in order.split(256):
                optimizer.zero_grad()
                loss = loss_function(model(x[indices], cal[indices]), y[indices])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total_loss += float(loss.detach()) * len(indices)
            entry = {'epoch': epoch+1, 'train_mae_step_Wh': total_loss/len(y)*ys}
            if valid is not None:
                model.eval()
                with torch.no_grad():
                    score = float(loss_function(model(vx, vc), vy))*ys
                entry['validation_mae_step_Wh'] = score
                if score < best - .01:
                    best, best_epoch, patience = score, epoch+1, 0
                else:
                    patience += 1
            history.append(entry)
            if valid is not None and patience >= 5:
                break
        state = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
        return {'state': state, 'x_mean': xm, 'x_std': xs, 'y_mean': ym, 'y_std': ys}, history, best_epoch

    _, history, best_epoch = train(inner_train, inner_val, max_epochs)
    # Sau khi chọn epoch, học lại từ đầu trên toàn TRAIN của outer fold.
    bundle, _, _ = train(train_mask, None, best_epoch)
    if device == 'cuda':
        torch.cuda.synchronize()
    bundle['training'] = {'device': device, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                          'torch': torch.__version__, 'seed': seed, 'epochs': best_epoch,
                          'max_epochs': max_epochs, 'inner_validation_start': str(cut),
                          'inner_train_n': int(inner_train.sum()), 'inner_val_n': int(inner_val.sum()),
                          'seconds': time.perf_counter()-started, 'history': history}
    return bundle


def predict_lstm(bundle, df, mask):
    import torch
    sequence, calendar = inputs(df)
    model = network().cpu()
    model.load_state_dict({key: torch.from_numpy(value.copy()) for key, value in bundle['state'].items()})
    model.eval()
    x = torch.from_numpy((sequence[mask]-bundle['x_mean'])/bundle['x_std'])
    cal = torch.from_numpy(calendar[mask])
    with torch.no_grad():
        batches = [model(x[i:i+1024], cal[i:i+1024]).numpy() for i in range(0, len(x), 1024)]
    return np.maximum(np.concatenate(batches)*bundle['y_std']+bundle['y_mean'], 0).astype(float)
