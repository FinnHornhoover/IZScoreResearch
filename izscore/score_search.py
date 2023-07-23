import sys
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Tuple, Union, TextIO
from argparse import ArgumentParser, Namespace

import sympy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pysr import PySRRegressor

from ff_metrics import mean_percentage_error


class DataTransform:
    @classmethod
    def normal(cls, val: np.ndarray, val_max: np.ndarray = None) -> np.ndarray:
        return val

    @classmethod
    def ratio(cls, val: np.ndarray, val_max: np.ndarray) -> np.ndarray:
        return np.true_divide(val, val_max, out=np.zeros_like(val), where=(~np.isclose(val_max, 0.0)))

    @classmethod
    def log(cls, val: np.ndarray, val_max: np.ndarray = None) -> np.ndarray:
        return np.log(val)

    @classmethod
    def exp(cls, val: np.ndarray, val_max: np.ndarray = None) -> np.ndarray:
        return np.exp(val)

    @classmethod
    def log_ratio(cls, val: np.ndarray, val_max: np.ndarray) -> np.ndarray:
        return cls.ratio(cls.log(val), cls.log(val_max))

    @classmethod
    def ratio_exp(cls, val: np.ndarray, val_max: np.ndarray) -> np.ndarray:
        return cls.exp(cls.ratio(val, val_max))


class InverseDataTransform:
    @classmethod
    def normal(cls, val: np.ndarray, val_max: np.ndarray = None) -> np.ndarray:
        return val

    @classmethod
    def ratio(cls, val: np.ndarray, val_max: np.ndarray) -> np.ndarray:
        return val * val_max

    @classmethod
    def log(cls, val: np.ndarray, val_max: np.ndarray = None) -> np.ndarray:
        return np.exp(val)

    @classmethod
    def exp(cls, val: np.ndarray, val_max: np.ndarray = None) -> np.ndarray:
        return np.log(val)

    @classmethod
    def log_ratio(cls, val: np.ndarray, val_max: np.ndarray) -> np.ndarray:
        return cls.log(cls.ratio(val, DataTransform.log(val_max)))

    @classmethod
    def ratio_exp(cls, val: np.ndarray, val_max: np.ndarray) -> np.ndarray:
        return cls.ratio(cls.exp(val), val_max)


class WeightTransform:
    @classmethod
    def normal(cls, data: np.ndarray, feature_names: List[str]) -> np.ndarray:
        return np.ones_like(data[:, feature_names.index('score')])

    @classmethod
    def target_scaled(cls, data: np.ndarray, feature_names: List[str]) -> np.ndarray:
        return 1. / data[:, feature_names.index('score')]

    @classmethod
    def max_score_scaled(cls, data: np.ndarray, feature_names: List[str]) -> np.ndarray:
        return 1. / data[:, feature_names.index('max_score')]


class WeightBracketExtractor:
    @classmethod
    def none(cls, row: Union[Tuple[int], List[int]], feature_names: List[str]) -> Union[None, Tuple[int]]:
        return None

    @classmethod
    def max_score(cls, row: Union[Tuple[int], List[int]], feature_names: List[str]) -> Union[None, Tuple[int]]:
        return (row[feature_names.index('max_score')], )

    @classmethod
    def max_score_pod(cls, row: Union[Tuple[int], List[int]], feature_names: List[str]) -> Union[None, Tuple[int]]:
        return (row[feature_names.index('max_score')], row[feature_names.index('pods')])

    @classmethod
    def max_score_time(cls, row: Union[Tuple[int], List[int]], feature_names: List[str]) -> Union[None, Tuple[int]]:
        return (row[feature_names.index('max_score')], row[feature_names.index('time')])

    @classmethod
    def max_score_pod_time(cls, row: Union[Tuple[int], List[int]], feature_names: List[str]) -> Union[None, Tuple[int]]:
        return (row[feature_names.index('max_score')], row[feature_names.index('pods')], row[feature_names.index('time')])


class DataDerivation:
    @classmethod
    def max_score_exp_ratios(cls, data: np.ndarray, feature_names: List[str], factor: float = 1.2) -> np.ndarray:
        max_score = data[:, feature_names.index('max_score')]
        max_pods = data[:, feature_names.index('max_pods')]
        max_time = data[:, feature_names.index('max_time')]
        pods = data[:, feature_names.index('pods')]
        time = data[:, feature_names.index('time')]

        return max_score * np.exp(factor * pods / max_pods - time / max_time)

    @classmethod
    def exp_pod_ratio(cls, data: np.ndarray, feature_names: List[str]) -> np.ndarray:
        return np.exp(data[:, feature_names.index('pods')] / data[:, feature_names.index('max_pods')])

    @classmethod
    def exp_time_ratio(cls, data: np.ndarray, feature_names: List[str]) -> np.ndarray:
        return np.exp(data[:, feature_names.index('time')] / data[:, feature_names.index('max_time')])


def encode_timestamp(timestamp: object) -> str:
    return str(timestamp).replace(' ', '_').replace(':', '')


def encode_iz_scope(iz_scope: str) -> str:
    return iz_scope.lower().replace(' ', '_').replace('(', '').replace(')', '')


def get_feature_information(args: Namespace) -> Tuple[List[int], List[int], List[Tuple[int, int]], List[str]]:
    static_feature_names = [
        'max_score',
        'max_pods',
        'max_time',
    ]
    derived_feature_names = args.derived_features or []
    dynamic_feature_names = [
        ('pods', 'max_pods'),
        ('time', 'max_time'),
        ('score', 'max_score'),
    ]
    ls = len(static_feature_names)
    lder = len(derived_feature_names)
    static_idx = list(range(ls))
    derived_idx = list(range(ls, ls + lder))
    dynamic_idx = [(ls + lder + i, static_feature_names.index(val_max)) for i, (_, val_max) in enumerate(dynamic_feature_names)]
    feature_names = static_feature_names + derived_feature_names + [val for val, _ in dynamic_feature_names]

    return static_idx, derived_idx, dynamic_idx, feature_names


def get_data(args: Namespace) -> Tuple[List[str], List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    with open(args.data_path) as r:
        score_dict = json.load(r)

    extract_bracket = getattr(WeightBracketExtractor, args.weight_bracket_mode)

    _, derived_idx, dynamic_idx, feature_names = get_feature_information(args)
    target_idx, target_max_idx = dynamic_idx[-1]

    iz_scope_list = []
    sym_feat_list = []
    sym_target_list = []
    sym_weight_list = []

    seen = set()
    weight_brackets = defaultdict(int)

    for k, score_info in score_dict.items():
        if args.disabled_izs is not None and k in args.disabled_izs:
            continue

        sym_data = []

        objs = score_info['scores']
        max_score = score_info['max_score']
        max_pods = score_info['max_pods']

        def append_record(**kwargs) -> None:
            list_record = [score_info.get(feat_name,
                                          kwargs.get(feat_name,
                                                     locals().get(feat_name, np.nan)))
                           for feat_name in feature_names]
            record = tuple(list_record)
            if record in seen:
                return
            seen.add(record)
            sym_data.append(list_record)
            weight_brackets[extract_bracket(record, feature_names)] += 1

        append_record(pods=max_pods, time=1, score=max_score)

        worst_max = sorted([o for o in objs if o['score'] == max_score], key=lambda d: d['pods'] - d['time'])
        if worst_max:
            o = worst_max[0]
            append_record(pods=o['pods'], time=1, score=max_score)
            append_record(pods=max_pods, time=o['time'], score=max_score)

        for o in objs:
            append_record(pods=o['pods'], time=o['time'], score=o['score'])

        sym_data_arr = np.array(sym_data, dtype=np.float32)

        # weight setting
        weight_transform = getattr(WeightTransform, args.weight_mode)
        sym_weight_arr = weight_transform(sym_data_arr, feature_names)
        sym_weight_bracket_arr = np.array([weight_brackets[extract_bracket(row, feature_names)] for row in sym_data], dtype=np.float32)
        sym_weight_arr /= sym_weight_bracket_arr
        sym_weight_arr[np.isclose(sym_data_arr[:, target_max_idx], sym_data_arr[:, target_idx])] *= args.max_score_weight_dampening
        sym_weight_arr /= sym_weight_arr.sum()

        # derived features
        for val_idx in derived_idx:
            deriver = getattr(DataDerivation, feature_names[val_idx])
            sym_data_arr[:, val_idx] = deriver(sym_data_arr, feature_names)

        # feature transforms
        feature_transform = getattr(DataTransform, args.feature_mode)
        for val_idx, val_max_idx in dynamic_idx[:-1]:
            sym_data_arr[:, val_idx] = feature_transform(sym_data_arr[:, val_idx], sym_data_arr[:, val_max_idx])

        # target transform
        target_transform = getattr(DataTransform, args.target_mode)
        sym_data_arr[:, target_idx] = target_transform(sym_data_arr[:, target_idx], sym_data_arr[:, target_max_idx])

        iz_scope_list.append(k)
        sym_feat_list.append(sym_data_arr[:, :target_idx])
        sym_target_list.append(sym_data_arr[:, target_idx])
        sym_weight_list.append(sym_weight_arr)

    return feature_names[:target_idx], iz_scope_list, sym_feat_list, sym_target_list, sym_weight_list


def make_model(args: Namespace, timestamp: object, iz_scope: str) -> Tuple[str, PySRRegressor]:
    equation_file = args.save_path.format(
        timestamp=encode_timestamp(timestamp),
        iz_scope=encode_iz_scope(iz_scope),
    )
    equation_dir = Path(equation_file).parent
    equation_dir.mkdir(parents=True, exist_ok=True)

    with open(equation_dir / 'args.json', 'w') as w:
        json.dump(vars(args), w, indent=4)

    print(args)

    full_objective = """
    function my_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        prediction, grad, complete = eval_grad_tree_array(tree, dataset.X, options; variable=true)

        if !complete
            return L(Inf)
        end

        # pred_fixed = prediction
        # loss = sum( abs((pred_fixed .- dataset.y) ./ dataset.X[1, :]) ) / dataset.n

        pred_fixed = round.(min.(dataset.X[1, :], prediction))
        loss = sum( ((pred_fixed .- dataset.y) ./ dataset.X[1, :]) .^ 2 ) / dataset.n

        # The "grad" is a Julia array which contains the gradient with shape (num_features, num_rows)
        # e.g., loss += sum(grad)
        if isnothing(grad)
            return loss
        end

        grad_loss = sum(grad[5, :]) - sum(grad[4, :])
        loss += 0.01 * grad_loss

        return loss
    end
    """
    full_objective = """
    function my_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        prediction, complete = eval_tree_array(tree, dataset.X, options)

        if !complete
            return L(Inf)
        end

        max_scores = dataset.X[1, :]
        exp_pred = exp.(prediction)
        diff = abs.(exp_pred .- dataset.y)
        diff_filt = diff .* ( ( (dataset.y .!= max_scores) .| (exp_pred .< max_scores) ) .& (diff .>= 0.5f0) )

        loss = sum( (diff_filt) .* dataset.weights ) / dataset.n

        return loss
    end
    """
    full_objective = """
    function my_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        prediction, complete = eval_tree_array(tree, dataset.X, options)
        !complete && return L(Inf)

        max_scores = dataset.X[1, :]
        clip_pred = min.(floor.(prediction), max_scores)
        diff = abs.(clip_pred .- dataset.y)

        loss = sum( diff .* dataset.weights ) / sum(dataset.weights)

        return loss
    end
    """
    return equation_file, PySRRegressor(
        model_selection='best',
        binary_operators=[
            #'+',
            '*',
            #'-',
            #'/',
            '^',
            'min',
            #'mod',
            #'idiv(a, b) = (a - mod(a, b)) / b',
        ],
        unary_operators=[
            #'sin',
            #'cos',
            #'log',
            #'log2',
            #'log10',
            #'sqrt',
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
            'floor',
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
            #'sigm': lambda x: (1.0 + sympy.tanh(x / 2.0)) / 2.0,
            #'exp2': lambda x: (2.0 ** x),
            #'exp10': lambda x: (10.0 ** x),
            #'herm': lambda x: x * x * (3. - 2. * x),
            #'floor': sympy.floor,
            #'ceil': sympy.ceiling,
            # 'sink': lambda x: sympy.sin(x * 3.14159274 * 0.5),
            #'clp': lambda x: sympy.Min(sympy.Max(x, 0.0), 1.0),
            # 'bee': lambda x: (sympy.sin(x * 3.14159274 * (0.2 + 2.5 * (x ** 3))) * ((1.0 - x) ** 2.2) + x) * (1.0 + 1.2 * (1.0 - x)),
        },
        loss='L1DistLoss()',
        #loss='L2DistLoss()',
        #full_objective=full_objective,
        early_stop_condition='f(loss, complexity) = (loss < 1e-12) && (complexity < 15)',
        constraints={
            'pow': (-1, 1),
            'exp': 5,
            #'sqrt': 5,
            #'log': 5,
            #'decay': (-1, 1),
            'min': (-1, 1),
            #'max': (-1, 1),
        },
        #complexity_of_variables=2,
        #complexity_of_constants=2,
        nested_constraints={
            #'sin': {'sin': 0, 'cos': 0},
            #'cos': {'sin': 0, 'cos': 0},
            #'log': {'exp': 1, 'log': 1, 'sqrt': 1},
            #'log': {'log': 0, 'floor': 0, 'min': 0},
            #'exp': {'exp': 0, 'pow': 0},
            'pow': {'pow': 0},#, 'exp': 0},
            'min': {'min': 0},
            'floor': {'min': 0, 'floor': 0},
            #'ceil': {'min': 0, 'ceil': 0},
            #'log': {'log': 1},
            #'sqrt': {'exp': 1, 'log': 1, 'sqrt': 1},
            #'tanh': {'tanh': 0},
            #'sigm': {'sigm': 0},
        },
        niterations=3000,
        #parsimony=0.0001,
        weight_optimize=0.1,
        #weight_mutate_operator=1.5,
        maxsize=30,
        adaptive_parsimony_scaling=200.0,
        warmup_maxsize_by=0.1,
        ncyclesperiteration=40,
        #population_size=75,  # default 33
        #tournament_selection_n=23,  # default 10
        #tournament_selection_p=0.8,  # default 0.86
        #fraction_replaced_hof=0.08,  # default 0.035
        #optimizer_iterations=25,  # default 8
        #crossover_probability=0.12,  # default 0.066
        #populations=24,  # default 15
        turbo=True,
        equation_file=equation_file,
        precision=64,
        #batching=True,
        #batch_size=32,
    )


def show_discovery_results(args: Namespace, save_path: str, model: PySRRegressor, iz_scope: str, sym_feat: np.ndarray, sym_target: np.ndarray, sym_weight: np.ndarray, visualize: bool = False) -> None:
    inverse_transform = getattr(InverseDataTransform, args.target_mode)

    static_idx, derived_idx, dynamic_idx, feature_names = get_feature_information(args)
    _, target_max_idx = dynamic_idx[-1]

    def print_stats(scaled_pred: np.ndarray, scaled_target: np.ndarray, signed_eps_ins_errors: np.ndarray, file: TextIO = sys.stdout) -> None:
        eps_ins_errors = np.abs(signed_eps_ins_errors)

        print('\n\n\n\n', file=file)
        print(iz_scope, 'Chosen Equation:', file=file)
        print(model.get_best().equation, file=file)
        print('+/- Relevant Errors:', signed_eps_ins_errors[eps_ins_errors > 0.0], file=file)
        print('Error Counts:', np.sum(eps_ins_errors > 0.0), '/', eps_ins_errors.shape[0], f'({100.0 * np.mean(eps_ins_errors > 0.0):.3f} %)', file=file)
        print(file=file)
        print('Mean Errors:', np.mean(eps_ins_errors), file=file)
        print('Mean Errors (Target Scaled):', np.mean(eps_ins_errors / scaled_target), file=file)
        print('Mean Errors (Max Score Scaled):', np.mean(eps_ins_errors / max_target), file=file)
        print(file=file)
        print('MAPE:', mean_percentage_error(scaled_pred, scaled_target, w=1), file=file)
        print('MAPEW:', mean_percentage_error(scaled_pred, scaled_target, w=sym_weight), file=file)
        print('MSPE:', mean_percentage_error(scaled_pred, scaled_target, w=1, ord=2), file=file)
        print('MSPEW:', mean_percentage_error(scaled_pred, scaled_target, w=sym_weight, ord=2), file=file)
        print(file=file)

        if file == sys.stdout:
            with open(Path(save_path).parent / 'log.txt', 'a') as a:
                print_stats(scaled_pred, scaled_target, signed_eps_ins_errors, file=a)

    def print_error(error: Exception, file: TextIO = sys.stdout) -> None:
        print('Error:', error, file=file)
        print(model.get_best().equation, 'contains faulty SymPy bindings, skipping...', file=file)

        if file == sys.stdout:
            with open(Path(save_path).parent / 'log.txt', 'a') as a:
                print_error(e, file=a)

    try:
        pred = model.predict(sym_feat)

        max_target = sym_feat[:, target_max_idx]
        scaled_target = inverse_transform(sym_target, max_target)
        scaled_pred = inverse_transform(pred, max_target)

        ignored_errors = (np.abs(scaled_pred - scaled_target) < 0.5) | (np.isclose(scaled_target, max_target) & (scaled_pred > max_target))
        scaled_pred[ignored_errors] = scaled_target[ignored_errors]
        signed_eps_ins_errors = scaled_pred - scaled_target

        print_stats(scaled_pred, scaled_target, signed_eps_ins_errors)

        if visualize:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            pp, tt = np.meshgrid(*[np.unique(sym_feat[:, val_idx]) for val_idx, _ in dynamic_idx[:2]])
            feat_all = np.stack((
                *[np.ones((1, *pp.shape)) * sym_feat[0, val_idx] for val_idx in static_idx],
                *[np.ones((1, *pp.shape)) * np.nan for _ in derived_idx],
                pp[np.newaxis, :, :],
                tt[np.newaxis, :, :],
            ), axis=0).reshape(len(static_idx) + len(derived_idx) + 2, -1).T

            for val_idx in derived_idx:
                deriver = getattr(DataDerivation, feature_names[val_idx])
                feat_all[:, val_idx] = deriver(feat_all, feature_names)

            pred_all = model.predict(feat_all)
            scaled_pred_all = inverse_transform(pred_all, max_target)

            largest_diff = np.abs(signed_eps_ins_errors).max()
            color_norm = mcolors.Normalize(vmin=-largest_diff, vmax=largest_diff)
            sc = ax.scatter(*[sym_feat[:, val_idx] for val_idx, _ in dynamic_idx[:2]], scaled_target, s=3, cmap='RdBu', c=signed_eps_ins_errors, norm=color_norm, depthshade=False)
            ax.plot_surface(pp, tt, scaled_pred_all.reshape(*pp.shape), color='white', alpha=0.4)
            fig.colorbar(sc, ax=ax)

            ax.set_title(iz_scope)
            ax.set_xlabel('pods')
            ax.set_ylabel('time')
            ax.set_zlabel('score')

            plt.show()

    except Exception as e:
        print_error(e)


def get_trained_discovery(args: Namespace, test_path: str, resume_path: str, iz_scope: str, timestamp: object, feature_names: str, sym_feat: np.ndarray, sym_target: np.ndarray, sym_weight: np.ndarray) -> Tuple[str, PySRRegressor]:
    print(iz_scope, 'Features Shape:', sym_feat.shape)

    if test_path:
        save_path = test_path.format(iz_scope=encode_iz_scope(iz_scope))
        model = PySRRegressor.from_file(save_path)
    elif resume_path:
        save_path = resume_path.format(iz_scope=encode_iz_scope(iz_scope))
        model = PySRRegressor.from_file(save_path, warm_start=True)
    else:
        save_path, model = make_model(args, timestamp, iz_scope)

    if not test_path:
        model.fit(sym_feat, sym_target, weights=sym_weight, variable_names=feature_names)

    return save_path, model


def discover(cmd_args: Namespace) -> None:
    args = cmd_args
    if cmd_args.test_path:
        with open(Path(cmd_args.test_path).parent / 'args.json') as r:
            args = Namespace(**json.load(r))

    feature_names, iz_scope_list, sym_feat_list, sym_target_list, sym_weight_list = get_data(args)
    timestamp = datetime.now()

    if args.fit_type == 'all':
        iz_scope_all = 'All IZs'
        sym_feat_all = np.concatenate(sym_feat_list, axis=0)
        sym_target_all = np.concatenate(sym_target_list, axis=0)
        sym_weight_all = np.concatenate(sym_weight_list, axis=0)

        save_path, model = get_trained_discovery(args, cmd_args.test_path, cmd_args.resume_path, iz_scope_all, timestamp, feature_names, sym_feat_all, sym_target_all, sym_weight_all)

        for iz_scope, sym_feat, sym_target, sym_weight in zip(iz_scope_list, sym_feat_list, sym_target_list, sym_weight_list):
            show_discovery_results(args, save_path, model, iz_scope, sym_feat, sym_target, sym_weight, visualize=cmd_args.visualize)

        show_discovery_results(args, save_path, model, iz_scope_all, sym_feat_all, sym_target_all, sym_weight_all)
    else:
        for iz_scope, sym_feat, sym_target, sym_weight in zip(iz_scope_list, sym_feat_list, sym_target_list, sym_weight_list):

            save_path, model = get_trained_discovery(args, cmd_args.test_path, cmd_args.resume_path, iz_scope, timestamp, feature_names, sym_feat, sym_target, sym_weight)

            show_discovery_results(args, save_path, model, iz_scope, sym_feat, sym_target, sym_weight, visualize=cmd_args.visualize)


def parse_args() -> Namespace:
    parser = ArgumentParser(description='FF Score Function Searcher.')
    parser.add_argument('--data-path', dest='data_path', type=str, default='data/merged_scores.json')
    parser.add_argument('--save-path', dest='save_path', type=str, default='score_hof/{timestamp}/hall_of_fame_{iz_scope}.csv')
    parser.add_argument('--test-path', dest='test_path', type=str, default='')
    parser.add_argument('--resume-path', dest='resume_path', type=str, default='')
    parser.add_argument('--fit-type', dest='fit_type', type=str, default='all',
                        choices=['all', 'per_iz'])
    parser.add_argument('--derived-features', dest='derived_features', type=str, nargs='+',
                        choices=[m for m in dir(DataDerivation) if not m.startswith('__')])
    parser.add_argument('--feature-mode', dest='feature_mode', type=str, default='normal',
                        choices=[m for m in dir(DataTransform) if not m.startswith('__')])
    parser.add_argument('--target-mode', dest='target_mode', type=str, default='normal',
                        choices=[m for m in dir(DataTransform) if not m.startswith('__')])
    parser.add_argument('--weight-mode', dest='weight_mode', type=str, default='normal',
                        choices=[m for m in dir(WeightTransform) if not m.startswith('__')])
    parser.add_argument('--weight-bracket-mode', dest='weight_bracket_mode', type=str, default='none',
                        choices=[m for m in dir(WeightBracketExtractor) if not m.startswith('__')])
    parser.add_argument('--max-score-weight-dampening', dest='max_score_weight_dampening', type=float, default=1.0)
    parser.add_argument('--disabled-izs', dest='disabled_izs', type=str, nargs='+')
    parser.add_argument('--visualize', action='store_true', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    discover(parse_args())
