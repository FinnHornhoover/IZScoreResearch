import json

import torch
from torch import nn, optim
from torch.nn.parameter import Parameter


class InsensitiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y, w):
        diff = torch.abs(y_pred - y)
        diff[diff < 0.5] = 0.0
        return (diff * w).sum() / w.sum()


class Prot1(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = Parameter(torch.tensor(1.0, dtype=torch.float64), requires_grad=True)

    def forward(self, x):
        max_score, max_pods, max_time, score_per_pod, ms, pr, pods_collected = torch.tensor_split(x, 7, dim=1)
        return (pods_collected * pr * torch.exp(ms - 1) + self.c) / max_pods - .5


def get_data():
    with open('data/merged_scores.json') as r:
        score_dict = json.load(r)

    with open('data/params.json') as r:
        params = json.load(r)

    sym_data = []
    sym_target = []
    sym_weights = []

    mss = {o['iz_name']: o['MS Multiplier'] for o in params}
    prs = {o['iz_name']: o['PR Multiplier'] for o in params}

    for k, score_info in score_dict.items():
        objs = [v for v in score_info['scores'] if 'fm' in v]
        max_score = score_info['max_score']
        max_pods = score_info['max_pods']
        max_time = score_info['max_time']

        sym_data.append([
            float(max_score),
            float(max_pods),
            float(max_time),
            max_score / max_pods,
            mss[k],
            prs[k],
            0.0,
        ])
        sym_target.append(0.0)
        sym_weights.append(1.0)

        for o in objs:
            sym_data.append([
                float(max_score),
                float(max_pods),
                float(max_time),
                max_score / max_pods,
                mss[k],
                prs[k],
                float(o['pods']),
            ])
            sym_target.append(float(o['fm']))
            sym_weights.append(1.0)

    sym_data = torch.DoubleTensor(sym_data)
    sym_target = torch.DoubleTensor(sym_target)
    sym_weights = torch.DoubleTensor(sym_weights)

    return sym_data, sym_target, sym_weights


def main():
    model = Prot1()

    sym_data, sym_target, sym_weights = get_data()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.StepLR(optimizer, step_size=7000, gamma=0.01)
    crit = InsensitiveLoss()

    model.eval()
    with torch.no_grad():
        optimizer.zero_grad()

        y_pred = model(sym_data).squeeze(-1).detach().cpu()
        y = sym_target.detach().cpu()
        w = sym_weights.detach().cpu()

    print('Prev Loss:', crit(y_pred, y, w).item())

    model.train()
    for epoch in range(10000):
        optimizer.zero_grad()

        pred = model(sym_data).squeeze(-1)
        loss = crit(pred, sym_target, sym_weights)

        print(f'Epoch: {epoch + 1:6d} Loss {loss.detach().cpu().item():.7f}', end='\r')

        loss.backward()
        optimizer.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        optimizer.zero_grad()

        y_pred = model(sym_data).squeeze(-1).detach().cpu()
        y = sym_target.detach().cpu()
        w = sym_weights.detach().cpu()

    print()
    print([abs(a - b) for a, b in zip(y_pred.numpy(), y.numpy()) if abs(round(a - b)) > 0])
    print('Errors', sum([abs(round(a - b)) > 0 for a, b in zip(y_pred.numpy(), y.numpy())]), 'out of', y_pred.shape[0])
    print('Loss:', crit(y_pred, y, w).item())
    print('Coef:', model.c.item())


if __name__ == '__main__':
    main()
