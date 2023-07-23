import json
from datetime import datetime

import sympy
import numpy as np
from pysr import PySRRegressor

from ff_metrics import mean_percentage_error


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
            max_score // max_pods,
            np.exp(mss[k] - 1),
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
                max_score // max_pods,
                np.exp(mss[k] - 1),
                prs[k],
                float(o['pods']),
            ])
            sym_target.append(float(o['fm']))
            sym_weights.append(1.0)

    sym_data = np.array(sym_data, dtype=np.float64)
    sym_target = np.array(sym_target, dtype=np.float64)
    sym_weights = np.array(sym_weights, dtype=np.float64)

    return sym_data, sym_target, sym_weights


def main():
    full_objective = """
    function my_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        prediction, complete = eval_tree_array(tree, dataset.X, options)

        if !complete
            return L(Inf)
        end

        diff = abs.((floor.(prediction .* dataset.X[end, :])) .- dataset.y)
        diff_filt = diff

        loss = sum( diff_filt .* dataset.weights ) / sum(dataset.weights)

        return loss
    end
    """
    model = PySRRegressor(
        model_selection='best',
        binary_operators=['+', '*', '-', '/', 'max', 'min'],
        unary_operators=[
            'sin',
            #'cos',
            'log',
            #'square',
            #'cube',
            #'quart(x) = x ^ 4.0f3',
            #'quint(x) = x ^ 5.0f3',
            #'log2',
            #'log10',
            #'sqrt',
            #'cbrt(x) = x ^ (1 / 3)',
            'exp',
            #'exp2',
            #'exp10',
            # 'floor',
            #'sinh',
            #'tanh',
            #'gamma',
            #'erf',
            #'floor',
            #'ceil',
            #'herm(x) = x * x * (3.0f3 - 2.0f3 * x)',
            # 'sink(x) = sin(x * 3.14159274f3 * 0.5f3)',
            #'clp(x) = min(1.0f3, max(0.0f3, x))',
            # 'bee(x) = (sin(x * 3.14159274f3 * (0.2f3 + 2.5f3 * (x ^ 3))) * ((1.0f3 - x) ^ 2.2f3) + x) * (1.0f3 + 1.2f3 * (1.0f3 - x))',
        ],
        extra_sympy_mappings={
            'min': lambda a, b: sympy.Piecewise((a, a < b), (b, True)),
            'max': lambda a, b: sympy.Piecewise((b, a < b), (a, True)),
            #'quart': lambda x: x ** 4.,
            #'quint': lambda x: x ** 5.,
            #'cbrt': lambda x: x ** (1. / 3.),
            #'herm': lambda x: x * x * (3. - 2. * x),
            #'exp2': lambda x: (2.0 ** x),
            #'exp10': lambda x: (10.0 ** x),
            # 'floor': sympy.floor,
            #'ceil': sympy.ceiling,
            # 'sink': lambda x: sympy.sin(x * 3.14159274 * 0.5),
            #'clp': lambda x: sympy.Min(1.0, sympy.Max(0.0, x)),
            # 'bee': lambda x: (sympy.sin(x * 3.14159274 * (0.2 + 2.5 * (x ** 3))) * ((1.0 - x) ** 2.2) + x) * (1.0 + 1.2 * (1.0 - x)),
        },
        #loss='L1EpsilonInsLoss(0.5f0)',
        full_objective=full_objective,
        constraints={
            'pow': (-1, 5),
            'exp': 5,
            #'exp2': 3,
            #'exp10': 3,
            #'sqrt': 3,
            #'log': 3,
            #'decay': (-1, 1),
            #'min': (-1, 1),
            #'max': (-1, 1),
        },
        nested_constraints={
            #'sin': {'sin': 0, 'cos': 0},
            #'cos': {'sin': 0, 'cos': 0},
        },
        niterations=300,
        #parsimony=0.00005,
        weight_optimize=0.05,
        #select_k_features=2,
        maxsize=30,
        adaptive_parsimony_scaling=200.0,
        warmup_maxsize_by=0.7,
        ncyclesperiteration=1200,
        #population_size=75,  # default 33
        #tournament_selection_n=23,  # default 10
        #tournament_selection_p=0.8,  # default 0.86
        #fraction_replaced_hof=0.08,  # default 0.035
        #optimizer_iterations=25,  # default 8
        #crossover_probability=0.12,  # default 0.066
        #populations=50,  # default 15
        turbo=True,
        equation_file=f"fm_hof/hall_of_fame_{str(datetime.now()).replace(' ', '_').replace(':', '')}.csv",
        precision=64,
    )

    sym_data, sym_target, sym_weights = get_data()
    model.fit(sym_data, sym_target, weights=sym_weights, variable_names=[
        'max_score',
        'max_pods',
        'max_time',
        'score_per_pod',
        'score_per_pod_idiv',
        'ms',
        'pr',
        'pods_collected',
    ])

    print(model)
    print(model.get_best().equation)

    preds = model.predict(sym_data) * sym_data[:, -1]
    print('Significant Diffs:', json.dumps([{'diff': abs(a - b), 'pred': a, 'target': b} for a, b in zip(preds, sym_target) if abs(np.floor(a) - b) > 0], indent=2))
    print('Errors', sum([abs(np.floor(a) - b) > 0 for a, b in zip(preds, sym_target)]), 'out of', preds.shape[0])
    print('MAPE:', np.mean([mean_percentage_error(a, b, w=1, safe=True) for a, b in zip(preds, sym_target)]))
    print('MAPEW:', np.mean([mean_percentage_error(a, b, w=w, safe=True) for a, b, w in zip(preds, sym_target, sym_weights)]))
    print('MSPE:', np.mean([mean_percentage_error(a, b, w=1, safe=True, ord=2) for a, b in zip(preds, sym_target)]))
    print('MSPEW:', np.mean([mean_percentage_error(a, b, w=w, safe=True, ord=2) for a, b, w in zip(preds, sym_target, sym_weights)]))


if __name__ == '__main__':
    main()
