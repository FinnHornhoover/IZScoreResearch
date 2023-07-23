import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import sympy
import matplotlib.pyplot as plt
from pysr import PySRRegressor


def main():
    with open('data/params.json') as r:
        data = json.load(r)

    df = pd.DataFrame.from_records(data)

    equation_path = Path('param_hof/' + str(datetime.now()).replace(' ', '_').replace(':', '') + '.csv')
    equation_path.parent.mkdir(parents=True, exist_ok=True)
    model = PySRRegressor(
        model_selection='best',
        binary_operators=[
            '+',
            '*',
            '-',
            '/',
            #'^',
            'max',
            'min',
            #'mod',
            'imean(x, y) = x * y / (x + y)',
            #'idiv(a, b) = (a - mod(a, b)) / b',
        ],
        unary_operators=[
            #'sin',
            #'cos',
            'log',
            #'log2',
            #'log10',
            'sqrt',
            #'square',
            #'cube',
            #'exp',
            #'min1(x) = min(x, 1.0f3)',
            #'exp2',
            #'exp10',
            #'floor',
            #'cosh',
            #'sinh',
            #'tanh',
            #'atanh_clip',
            #'sigm(x) = (1.0f3 + tanh(x / 2.0f3)) / 2.0f3',
            #'logit(x) = log(x) / log(1 - x)',
            #'floor',
            #'ceil',
            #'herm(x) = x * x * (3.0f3 - 2.0f3 * x)',
            #'sink(x) = sin(x * 3.14159274f3 * 0.5f3)',
            #'clp(x) = min(max(x, 0.0f3), 1.0f3)',
            #'bee(x) = (sin(x * 3.14159274f3 * (0.2f3 + 2.5f3 * (x ^ 3))) * ((1.0f3 - x) ^ 2.2f3) + x) * (1.0f3 + 1.2f3 * (1.0f3 - x))',
        ],
        extra_sympy_mappings={
            'min': lambda a, b: sympy.Piecewise((a, a < b), (b, True)),
            'max': lambda a, b: sympy.Piecewise((b, a < b), (a, True)),
            #'idiv': lambda a, b: (a - (a % b)) / b,
            #'min1': lambda x: sympy.Piecewise((x, x < 1.0), (1.0, True)),
            'sigm': lambda x: (1.0 + sympy.tanh(x / 2.0)) / 2.0,
            #'logit': lambda x: sympy.log(x) / sympy.log(1.0 - x),
            #'exp2': lambda x: (2.0 ** x),
            #'exp10': lambda x: (10.0 ** x),
            'herm': lambda x: x * x * (3. - 2. * x),
            'imean': lambda a, b: a * b / (a + b),
            #'floor': sympy.floor,
            #'ceil': sympy.ceiling,
            # 'sink': lambda x: sympy.sin(x * 3.14159274 * 0.5),
            #'clp': lambda x: sympy.Min(sympy.Max(x, 0.0), 1.0),
            # 'bee': lambda x: (sympy.sin(x * 3.14159274 * (0.2 + 2.5 * (x ** 3))) * ((1.0 - x) ** 2.2) + x) * (1.0 + 1.2 * (1.0 - x)),
        },
        loss='L1DistLoss()',
        #loss='L2DistLoss()',
        #full_objective=full_objective,
        #early_stop_condition='f(loss, complexity) = (loss < 1e-12) && (complexity < 15)',
        constraints={
            'pow': (-1, 5),
            'exp': 5,
            #'sqrt': 5,
            #'log': 5,
            #'decay': (-1, 1),
            'min': (-1, 1),
            'max': (-1, 1),
        },
        nested_constraints={
            #'sin': {'sin': 0, 'cos': 0},
            #'cos': {'sin': 0, 'cos': 0},
            #'log': {'exp': 1, 'log': 1, 'sqrt': 1},
            #'log': {'log': 0, 'floor': 0, 'min': 0},
            #'exp': {'exp': 0},# 'pow': 0},
            #'pow': {'pow': 0, 'exp': 0},
            #'min': {'min': 0},
            #'floor': {'min': 0, 'floor': 0},
            #'ceil': {'min': 0, 'ceil': 0},
            'log': {'log': 1},
            #'sqrt': {'exp': 1, 'log': 1, 'sqrt': 1},
            #'tanh': {'tanh': 0},
            #'sigm': {'sigm': 0},
        },
        niterations=6000,
        #parsimony=0.0001,
        weight_optimize=0.1,
        #complexity_of_constants=2,
        #weight_mutate_operator=1.5,
        maxsize=30,
        adaptive_parsimony_scaling=200.0,
        warmup_maxsize_by=0.7,
        #ncyclesperiteration=1200,
        #population_size=75,  # default 33
        #tournament_selection_n=23,  # default 10
        #tournament_selection_p=0.8,  # default 0.86
        #fraction_replaced_hof=0.08,  # default 0.035
        #optimizer_iterations=25,  # default 8
        #crossover_probability=0.12,  # default 0.066
        #populations=24,  # default 15
        turbo=True,
        equation_file=str(equation_path),
        precision=64,
        #batching=True,
        #batch_size=32,
    )

    model.fit(df[['max_score', 'max_pods', 'max_time', 'epid', 'level']], df['scale_factor'])

    print('Chosen Equation:')
    print(model.get_best().equation)
    print(df['scale_factor'].tolist())
    print(model.predict(df[['max_score', 'max_pods', 'max_time', 'epid', 'level']]))

    model.fit(df[['max_score', 'max_pods', 'max_time', 'epid', 'level']], df['pod_factor'])

    print('Chosen Equation:')
    print(model.get_best().equation)
    print(df['pod_factor'].tolist())
    print(model.predict(df[['max_score', 'max_pods', 'max_time', 'epid', 'level']]))


if __name__ == '__main__':
    main()
