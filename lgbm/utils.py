from lightgbm import callback, Booster, Dataset
from lightgbm.compat import string_type, integer_types, range_, zip_
from lightgbm.engine import _make_n_folds, _agg_cv_result
import copy
import collections
from operator import attrgetter

#change a bit original lightGBM cross validation method to see both train and test sample performance
def cv_changed(params, train_set, num_boost_round=100,
               folds=None, nfold=5, stratified=True, shuffle=True,
               metrics=None, fobj=None, feval=None, init_model=None,
               feature_name='auto', categorical_feature='auto',
               early_stopping_rounds=None, fpreproc=None,
               verbose_eval=None, show_stdv=True, seed=0,
               callbacks=None):
    if not isinstance(train_set, Dataset):
        raise TypeError("Traninig only accepts Dataset object")

    params = copy.deepcopy(params)
    if fobj is not None:
        params['objective'] = 'none'
    for alias in ["num_iterations", "num_iteration", "n_iter", "num_tree", "num_trees",
                  "num_round", "num_rounds", "num_boost_round", "n_estimators"]:
        if alias in params:
            warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))
            num_boost_round = params.pop(alias)
            break
    for alias in ["early_stopping_round", "early_stopping_rounds", "early_stopping"]:
        if alias in params:
            warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))
            early_stopping_rounds = params.pop(alias)
            break

    if num_boost_round <= 0:
        raise ValueError("num_boost_round should be greater than zero.")
    if isinstance(init_model, string_type):
        predictor = _InnerPredictor(model_file=init_model, pred_parameter=params)
    elif isinstance(init_model, Booster):
        predictor = init_model._to_predictor(dict(init_model.params, **params))
    else:
        predictor = None
    train_set._update_params(params) \
             ._set_predictor(predictor) \
             .set_feature_name(feature_name) \
             .set_categorical_feature(categorical_feature)

    if metrics is not None:
        params['metric'] = metrics

    results = collections.defaultdict(list)
    cvfolds = _make_n_folds(train_set, folds=folds, nfold=nfold,
                            params=params, seed=seed, fpreproc=fpreproc,
                            stratified=stratified, shuffle=shuffle)

    # setup callbacks
    if callbacks is None:
        callbacks = set()
    else:
        for i, cb in enumerate(callbacks):
            cb.__dict__.setdefault('order', i - len(callbacks))
        callbacks = set(callbacks)
    if early_stopping_rounds is not None:
        callbacks.add(callback.early_stopping(early_stopping_rounds, verbose=False))
    if verbose_eval is True:
        callbacks.add(callback.print_evaluation(show_stdv=show_stdv))
    elif isinstance(verbose_eval, integer_types):
        callbacks.add(callback.print_evaluation(verbose_eval, show_stdv=show_stdv))

    callbacks_before_iter = {cb for cb in callbacks if getattr(cb, 'before_iteration', False)}
    callbacks_after_iter = callbacks - callbacks_before_iter
    callbacks_before_iter = sorted(callbacks_before_iter, key=attrgetter('order'))
    callbacks_after_iter = sorted(callbacks_after_iter, key=attrgetter('order'))

    for i in range_(num_boost_round):
        for cb in callbacks_before_iter:
            cb(callback.CallbackEnv(model=cvfolds,
                                    params=params,
                                    iteration=i,
                                    begin_iteration=0,
                                    end_iteration=num_boost_round,
                                    evaluation_result_list=None))
        cvfolds.update(fobj=fobj)
        res = [_agg_cv_result(cvfolds.eval_valid(feval)), _agg_cv_result(cvfolds.eval_train(feval))]

        for _, key, mean, _, std in res[0]:
            results[key + '-val-mean'].append(mean)
            results[key + '-val-stdv'].append(std)
        for _, key, mean, _, std in res[1]:
            results[key + '-train-mean'].append(mean)
            results[key + '-train-stdv'].append(std)
            
        try:
            for cb in callbacks_after_iter:
                cb(callback.CallbackEnv(model=cvfolds,
                                        params=params,
                                        iteration=i,
                                        begin_iteration=0,
                                        end_iteration=num_boost_round,
                                        evaluation_result_list=res[0]))
        except callback.EarlyStopException as earlyStopException:
            cvfolds.best_iteration = earlyStopException.best_iteration + 1
            for k in results:
                results[k] = results[k][:cvfolds.best_iteration]
            break
    
    return dict(results)

