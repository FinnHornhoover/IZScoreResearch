import json
from collections import defaultdict

import pandas as pd
import torch
from torch import nn, optim
from torch.nn.parameter import Parameter


class InsensitiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y_pred, y, w):
        max_score = x[:, 0]

        diff = torch.abs(y_pred - y)
        diff[diff < 0.5] = 0.0
        diff[torch.isclose(y, max_score) & (y_pred > max_score)] = 0.0

        return (diff * w).sum() / w.sum()


class Prot1(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = Parameter(torch.FloatTensor([-2.7010932]), requires_grad=True)
        self.c1 = Parameter(torch.FloatTensor([1.5081385]), requires_grad=True)
        self.c2 = Parameter(torch.FloatTensor([0.48413983]), requires_grad=True)

    def forward(self, x):
        max_score, max_pods, max_time, epid, pods_collected, time = torch.tensor_split(x, 6, dim=1)
        return torch.minimum(((((torch.maximum(torch.FloatTensor([0.0]), pods_collected + self.c0) / max_pods) ** self.c1) + self.c2) / torch.exp(time / max_time)) * max_score, max_score)


class Prot2(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = Parameter(torch.FloatTensor([-0.82721686, 0.0, 1.2334778, 2.4358647, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]), requires_grad=True)

    def forward(self, x):
        max_score, max_pods, max_time, epid, pods_collected, time = torch.tensor_split(x, 6, dim=1)
        to_exp = self.c[0] + self.c[9] * (pods_collected + self.c[1]) / (max_pods / self.c[2] + self.c[3]) - self.c[10] * (time + self.c[4]) / (max_time / self.c[5] + self.c[6])
        return self.c[7] + (max_score + self.c[8]) * torch.exp(to_exp)


class Prot3(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c = Parameter(torch.FloatTensor([0.1823799, -0.13071637, 0.4557713, 0.009702035, 3.407912067589078, 0.0, 1.0]), requires_grad=True)

    def forward(self, x):
        max_score, max_pods, max_time, epid, pods_collected, time = torch.tensor_split(x, 6, dim=1)
        return self.c[6] * ((torch.sqrt(torch.abs(self.c[0]) / (torch.sqrt(torch.exp((self.c[1] * pods_collected) / (time + self.c[2]) + self.c[3])) ** time))) ** torch.sigmoid(self.c[4]))


class Prot4(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = Parameter(torch.FloatTensor([1.0, 1.0, 0.0, 1.0, 0.0]), requires_grad=True)

    def forward(self, x):
        max_score, max_pods, max_time, epid, pods_collected, time = torch.tensor_split(x, 6, dim=1)
        return self.c[4] + self.c[3] * max_score * (torch.sigmoid(pods_collected * self.c[0] - time * self.c[1] + self.c[2]))


class Prot5(nn.Module):
    def __init__(self, scale_factor = 8.0, pod_factor = 1.2, time_factor = 1.0, grad_scale_factor = True, grad_pod_factor = True, grad_time_factor = False):
        super().__init__()
        self.scale_factor = Parameter(torch.tensor(scale_factor, dtype=torch.float64), requires_grad=grad_scale_factor)
        self.pod_factor = Parameter(torch.tensor(pod_factor, dtype=torch.float64), requires_grad=grad_pod_factor)
        self.time_factor = Parameter(torch.tensor(time_factor, dtype=torch.float64), requires_grad=grad_time_factor)

    def forward(self, x):
        max_score, max_pods, max_time, epid, pods_collected, time = torch.tensor_split(x, 6, dim=1)
        to_exp = self.pod_factor * (pods_collected / max_pods) - self.time_factor * (time / max_time) + self.scale_factor
        return torch.exp(to_exp) - .5


def get_data(use_ratios=False):
    with open('data/merged_scores.json') as r:
        score_dict = json.load(r)

    sym_data_list = []
    sym_target_list = []
    sym_weights_list = []

    weight_brackets = defaultdict(int)
    seen = set()

    for score_info in score_dict.values():
        objs = score_info['scores']
        max_score = score_info['max_score']
        max_pods = score_info['max_pods']
        max_time = score_info['max_time']
        epid = score_info['epid']

        sym_data = []
        sym_target = []

        sym_data.append([
            float(max_score),
            float(max_pods),
            float(max_time),
            float(epid),
            max_pods / (max_pods if use_ratios else 1.0),
            1.0 / (max_time if use_ratios else 1.0),
        ])
        sym_target.append(max_score)
        weight_brackets[(max_score, max_pods)] += 1

        worst_max = sorted([o for o in objs if o['score'] == max_score], key=lambda d: d['pods'] - d['time'])
        if worst_max:
            o = worst_max[0]
            sym_data.append([
                float(max_score),
                float(max_pods),
                float(max_time),
                float(epid),
                o['pods'] / (max_pods if use_ratios else 1.0),
                1.0 / (max_time if use_ratios else 1.0),
            ])
            sym_target.append(max_score)
            weight_brackets[(max_score, o['pods'])] += 1

            sym_data.append([
                float(max_score),
                float(max_pods),
                float(max_time),
                float(epid),
                max_pods / (max_pods if use_ratios else 1.0),
                o['time'] / (max_time if use_ratios else 1.0),
            ])
            sym_target.append(max_score)
            weight_brackets[(max_score, max_pods)] += 1

        for o in objs:
            record = (
                float(max_score),
                float(max_pods),
                float(max_time),
                float(epid),
                o['pods'] / (max_pods if use_ratios else 1.0),
                o['time'] / (max_time if use_ratios else 1.0),
            )

            if record in seen:
                continue

            seen.add(record)
            sym_data.append(list(record))
            sym_target.append(float(o['score']))
            weight_brackets[(max_score, o['pods'])] += 1

        sym_data_list.append(torch.DoubleTensor(sym_data))
        sym_target_list.append(torch.DoubleTensor(sym_target))
        sym_weights_list.append(torch.ones(len(sym_data), dtype=torch.float64))

    return list(score_dict), sym_data_list, sym_target_list, sym_weights_list


def main():
    params = []

    new_old_transfer = {}

    for k, sym_data, sym_target, sym_weights in zip(*get_data()):
        if k.endswith('(Old)'):
            model = Prot5(**new_old_transfer[k[:-6]])
        else:
            model = Prot5(grad_pod_factor=(k in [
                "Megas' Last Stand (F)",
                'Sweet Revenge',
                'Dizzy World',
                'Skypad Space Port',
                'Dinosaur Graveyard (New)',
                'Inferno Fields',
                'Dark Tree Clearing',
            ]))

        optimizer = optim.Adam(model.parameters(), lr=5e-3)
        sched = optim.lr_scheduler.StepLR(optimizer, step_size=7000, gamma=0.01)
        crit = InsensitiveLoss()

        model.eval()
        with torch.no_grad():
            optimizer.zero_grad()

            y_pred = model(sym_data).squeeze(-1).detach().cpu()
            y = sym_target.detach().cpu()
            w = sym_weights.detach().cpu()

        print('Prev Loss:', crit(sym_data, y_pred, y, w).item())

        model.train()
        for epoch in range(10000):
            optimizer.zero_grad()

            pred = model(sym_data).squeeze(-1)
            loss = crit(sym_data, pred, sym_target, sym_weights)

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
        print([abs(a - b) for s, a, b in zip(sym_data.numpy(), y_pred.numpy(), y.numpy()) if abs(round(a - b)) > 0 and a <= s[0]])
        print('Errors', sum([abs(round(a - b)) > 0 and a <= s[0] for s, a, b in zip(sym_data.numpy(), y_pred.numpy(), y.numpy())]), 'out of', y_pred.shape[0])
        print('Loss:', crit(sym_data, y_pred, y, w).item())
        print(k, list(model.parameters()))
        params.append({
            'iz_name': k,
            'max_score': int(sym_data[0, 0].item()),
            'max_pods': int(sym_data[0, 1].item()),
            'max_time': int(sym_data[0, 2].item()),
            'epid': int(sym_data[0, 3].item()),
            'scale_factor': model.scale_factor.item(),
            'pod_factor': model.pod_factor.item(),
            'time_factor': model.time_factor.item(),
        })
        print()

        if k.endswith('(New)'):
            new_old_transfer[k[:-6]] = {
                'scale_factor': model.scale_factor.item(),
                'pod_factor': model.pod_factor.item(),
                'time_factor': model.time_factor.item(),
                'grad_scale_factor': model.scale_factor.requires_grad,
                'grad_pod_factor': model.pod_factor.requires_grad,
                'grad_time_factor': model.time_factor.requires_grad,
            }

    with open('data/params.json', 'w') as w:
        json.dump(params, w, indent=4)

    pd.DataFrame.from_records(params).to_csv('data/params.csv', sep=';', index=False)


if __name__ == '__main__':
    main()
