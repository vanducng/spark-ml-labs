# functions for creating model and selection of hyperparameters
def objective_lgb(space, X, Y, cat_feats, NFALG_PRINT_HIST, NUM_FOLDS):
    #global iteration, best_auc_so_far, best_ntrees
    global iteration, best_auc_val_so_far, best_auc_val_std_so_far, best_ntrees, best_auc_train_so_far, best_auc_train_std_so_far
    iteration += 1   
    ### if NFALG_PRINT_HIST==False: print('Num step: {}'.format(iteration), end="\r")
    params = {'num_leaves': int(space['num_leaves']),
              'min_data_in_leaf': int(space['min_data_in_leaf']), 
              'max_depth': int(space['max_depth']),    
              'bagging_freq': int(space['bagging_freq']),    
              'max_bin': int(space['max_bin']),    
              
              'learning_rate': space['learning_rate'], 
              'feature_fraction': space['feature_fraction'],             
              'bagging_fraction': space['bagging_fraction'],
              'n_jobs': space['n_jobs'],
              'objective': space['objective'],
              'random_state': space['random_state']
             }

    ds = lgb.Dataset(X, 
                     Y, 
                     categorical_feature=cat_feats)
    
    cv_res = lgb.cv(params,
                    ds, 
                    metrics='auc',
                    num_boost_round=1000,
                    early_stopping_rounds=50,
                    nfold=len(time_folds) if time_folds else NUM_FOLDS,
                    folds=time_folds if time_folds else None,
                    categorical_feature=cat_feats)
    
    auc_val = cv_res['auc-val-mean'][-1]  #AUC  for the last boosting of Validation sample
    auc_val_std = cv_res['auc-val-stdv'][-1] #AUC STD for the last boosting of Validation sample
    auc_train = cv_res['auc-train-mean'][-1]  #AUC for the last boosting of Training Sample
    auc_train_std = cv_res['auc-train-stdv'][-1] #AUC STD for the last boosting of Training Sample
    
    # choosing best parameters based on a comparison between a new calculated and the last best
    if (((auc_val-2*auc_val_std)-0.5*abs((auc_train)-(auc_val))) > 
        ((best_auc_val_so_far-best_auc_val_std_so_far)-
         0.5*abs((best_auc_train_so_far)-(best_auc_val_so_far)))):
        best_auc_val_so_far = auc_val
        best_auc_val_std_so_far = auc_val_std
        best_ntrees = len(cv_res['auc-val-mean'])
        best_auc_train_so_far = auc_train
        best_auc_train_std_so_far = auc_train_std

    if NFALG_PRINT_HIST:
        print('iteration {}'.format(iteration))
        print("---AUC SCORE (val):\t{} (std: {}) ({} (std: {}) best so far)".format(round(auc_val,6), round(auc_val_std,6), 
                                                                                    round(best_auc_val_so_far,6), round(best_auc_val_std_so_far,6)))
        print("---AUC SCORE (train):\t{} (std: {}) ({} (std: {}) best so far)".format(round(auc_train,6), round(auc_train_std,6), 
                                                                                      round(best_auc_train_so_far,6), round(best_auc_train_std_so_far,6)))
        print("---ntrees:\t{} ({} best so far)\n".format(len(cv_res['auc-val-mean']), best_ntrees))
    
    return{'loss':1-((auc_val-auc_val_std)-0.5*abs((auc_train+auc_train_std)-(auc_val-auc_val_std))), 'status': STATUS_OK}

def run_hyperopt_lgb(x_train, y_train, x_test, y_test, cat_feats, evals=10, NFALG_PRINT_HIST=True, NUM_FOLDS = 3):
    iteration = 0
    
    params_space = {'num_leaves':[4, 12, 50, 100, 150, 200],
                    'min_data_in_leaf':[20, 50, 100, 250],
                    'max_depth':[2, 4, 6], # [2, 4, 6, 8, 10, 12, 14, 25]
                    'bagging_freq':[1, 2, 3, 5, 10],
                    'max_bin':[100, 255, 500],
                    
                    #'learning_rate':[0.01, 0.02, 0.03, 0.05, 0.1, 0.3, 0.5, 0.8, 1],
                    'learning_rate':[.01, .05, .1, .3, .5, .8],
                    'feature_fraction':[.25, .3, .5, .7],
                    'bagging_fraction':[.3, .5, .7],
                    'n_jobs': [8],
                    'objective': ['binary'], 
                    'random_state': [123]}
    # Variable for a final choice of the parameters (inputs) of the algorithm LightGBM
    space = {'num_leaves' : hp.choice('num_leaves', params_space['num_leaves']),
             'min_data_in_leaf' : hp.choice('min_data_in_leaf', params_space['min_data_in_leaf']),
             'max_depth' : hp.choice('max_depth', params_space['max_depth']),
             'bagging_freq' : hp.choice('bagging_freq', params_space['bagging_freq']),
             'max_bin' : hp.choice('max_bin', params_space['max_bin']),             
             
             'learning_rate' : hp.choice('learning_rate', params_space['learning_rate']),
             'feature_fraction' : hp.choice('feature_fraction', params_space['feature_fraction']),
             'bagging_fraction' : hp.choice('bagging_fraction', params_space['bagging_fraction']),
             'n_jobs' : hp.choice('n_jobs', params_space['n_jobs']),
             'objective' : hp.choice('objective', params_space['objective']),
             'random_state' : hp.choice('random_state', params_space['random_state'])}

    trials = Trials() # hyperOpt variable for choosing algorithm for best parameters finding
    # Running optimization for finding best parameters 
    best_params = fmin(fn=partial(objective_lgb, X=x_train, Y=y_train, cat_feats=cat_feats, 
                                  NFALG_PRINT_HIST=NFALG_PRINT_HIST, NUM_FOLDS=NUM_FOLDS),#objective
               space=space,
                algo=tpe.suggest, # Algorithm of finding optimal parameters. Please check how it works
                max_evals=evals, # number of iterations
                trials=trials) # object for Hyper Optimization

    print("--Best params metrics AUC SCORE:{} (std: {})  ntrees: {}".format(round(best_auc_val_so_far,6), round(best_auc_val_std_so_far,6), best_ntrees))
    
    for k in params_space.keys():  # востанавливает лучшие значения по их индексам
        best_params[k] = params_space[k][best_params[k]] # best params in a different format
            
    train = lgb.Dataset(x_train,
                        y_train,
                        categorical_feature=cat_feats) #  categorical_feature 
    
    test = lgb.Dataset(x_test,
                       y_test, 
                       categorical_feature=cat_feats)
    
    booster = lgb.train(best_params,   # calling the algorithm itself
                        train_set=train,
                        valid_sets=[train, test],
                        num_boost_round=best_ntrees,
                        categorical_feature=cat_feats,
                        verbose_eval=True)

    return booster, best_params