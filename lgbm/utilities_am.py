#import cx_Oracle as __orcl
import datetime as __datetime
import pandas as __pd
import numpy as __np
import matplotlib.pyplot as __plt
import seaborn as __sns
from tqdm import tqdm_notebook as __tqdm_notebook
from sklearn.preprocessing import StandardScaler as __StandardScaler
from sklearn.model_selection import (GridSearchCV as __GridSearchCV
                                     , StratifiedKFold as __StratifiedKFold
                                     , cross_val_score as __cross_val_score)
from sklearn.metrics import (roc_curve as __roc_curve
                             , auc as __auc
                             , roc_auc_score as __roc_auc_score
                             , average_precision_score as __average_precision_score
                             , precision_recall_curve as __precision_recall_curve
                             , accuracy_score as __accuracy_score
                             , classification_report as __classification_report
                             , confusion_matrix as __confusion_matrix
                             , make_scorer as __make_scorer)
#import xgboost as __xgb
from sklearn.linear_model import LogisticRegression as __LogisticRegression



__scoring = {'AUC': 'roc_auc', 'Accuracy': __make_scorer(__accuracy_score)}

iteration, best_auc_val_so_far, best_auc_val_std_so_far, best_ntrees, best_auc_train_so_far, best_auc_train_std_so_far = 0,0,0,0,0,0

def __get_oracle_datatypes_from_df(df):
    d = {
        'int64':'number',
        'int32':'number',
        'float64':'number',
        'float32':'number',
        'datetime64[ns]':'date',
        'object':'varchar(500)',
        'category':'varchar(500)'
        }

    dt = []

    for c in df.columns.values:
        # check if an object is actualy a date
        if str(df[c].dtype) == 'object' and isinstance(df.loc[0, c], __datetime.date):
            dt.append('date')
            continue

        dt.append(d[str(df[c].dtype)])

    return dt

def __get_create_table_str_from_df(df, table_name):
	cols_types = zip(df.columns.values, __get_oracle_datatypes_from_df(df))

	temp_arr = []
	for p in cols_types:
		temp_arr.append(p[0] + ' ' + p[1])

	cols_str = ', '.join(temp_arr)

	return 'create table {} ({})'.format(table_name, cols_str)

def __get_insert_into_table_str_from_df(df, table_name):
	cols = ', '.join(df.columns.values)

	temp_arr = []
	for i in range(df.shape[1]):
		temp_arr.append(':{}'.format(i))

	s = ', '.join(temp_arr)

	return '''insert --+ append
     into {} ({}) values ({})'''.format(table_name, cols, s)

def __check_columns(df):
    illegal_chars = '`~!@3=#$%^&*()"+_-=!";%:?*\{\}'
    for col in df.columns.values:
        for c in illegal_chars:
            if c in col:
                if c != '_' or col.find('_') == 0:

                    raise ValueError('Column {} contains illegal character {}'.format(col, c))

def __dbsave(connection, df, table_name, drop_if_exists, append):
    if append & drop_if_exists:
        raise ValueError('Is it drop or append? Make up your mind already.')
        return

    __check_columns(df)

    str_create = __get_create_table_str_from_df(df, table_name)
    str_insert = __get_insert_into_table_str_from_df(df, table_name)

    cursor = __orcl.Cursor(connection)

    try:
        cursor.execute(str_create)
    except Exception as e:
        if drop_if_exists:
            cursor.execute('drop table ' + table_name)
            cursor.execute(str_create)
        elif not append:
            raise ValueError('Table {} already exists. Set drop_if_exists=True to drop, or append=True to append existing table'.format(table_name))

    cursor.prepare(str_insert)
    cursor.executemany(None, list(df.values))
    connection.commit()
    cursor.close()

def dbsave(connection, df, table_name, drop_if_exists=False, append=False):
    """
    Writes DataFrame to oracle database.

    Parameters
    ----------
    connection : cx_Oracle.Connection
        Connection to oracle db.

    df : pandas.core.frame.DataFrame
        Dataframe to save.

    table_name : string
        Table name.

    drop_if_exists : boolean, default False
        Drop table if it already exists.
    """

    __dbsave(connection, df, table_name, drop_if_exists, append)

def get_feature_importances_df(X, model):
    """
	Returns DataFrame with model feature importances. Model has to have feature_importances_ attribute.

	Parameters
    ----------
	X: pandas.core.frame.DataFrame or list
	Either DataFrame with data on which model was fitted or list of columns

	model:	pandas.core.frame.DataFrame
	Any sklearn classifier with feature_importances_.
    """
    cols = []

    if isinstance(X, __pd.core.frame.DataFrame):
        cols = X.columns
    else:
        cols = X
    
    if hasattr(model, 'feature_importances_'):
        f_i = model.feature_importances_
    else:
        f_i = model.feature_importance()
    
    return __pd.DataFrame(list(zip(cols, f_i)),
                          columns=['feature','importance']).sort_values('importance',
                                                                        ascending=False).reset_index(drop=True)
def _cols_from_top_features(connection, tname, importances_df, top_n, verbose):
    sql = 'SELECT * FROM '+tname+' WHERE ROWNUM = 1'
    row = __pd.read_sql_query(sql, connection)

    all_cols = row.columns.values

    cols_to_load = set()
    dummy_cols = set()

    for col in importances_df[:top_n].feature.values:
        if col in all_cols:
            cols_to_load.add(col)
        else:
            correct_column_name = ''
            if col[:col.rfind('___')] == 'OF_CODE_SEGMENT_NEW2':
                correct_column_name = 'OF_CODE_SEGMENT_NEW'
            else:
                correct_column_name = row.filter(regex='.*'+col[:col.rfind('___')].strip('_')+'.*').columns.values[0]

            cols_to_load.add(correct_column_name)
            dummy_cols.add(correct_column_name)
            if verbose:
                print ('Column: {} is not found.\n{} passed instead\n\n'.format(col, correct_column_name))

    return	list(cols_to_load), list(dummy_cols)

def cols_from_top_features(connection, tname, importances_df, top_n=50, verbose=False):
	"""
    Takes DataFrame with feature_importances (get_feature_importances_df(X, model)) and returns columns to read from DWH.

    Parameters
    -------------------
    tname: String
    Table name in DWH from which you're intended to read the data.

    importances_df: DataFrame
    DataFrame with feature importances.

    top_n: int, default=50
    Number of top features to use.

    verbose: boolean, default=False
    Whether to print trace information or not.

    Returns
    -------------------
    (list of all columns to read from DWH, list of columns required to be transformed to dummy)
    """
	return _cols_from_top_features(connection, tname, importances_df, top_n, verbose)

def get_lift_df(pred, y_true, bins=10):
    """
    Returns a Pandas DataFrame with the average lift generated by the model in each bin

    Parameters
    -------------------
    pred: list
    Predicted probabilities

    y_true: list
    Real target values

    bins: int, default=10
    Number of equal sized buckets to divide observations across
    """

    cols = ['pred', 'actual']
    data = [pred, y_true]
    df = __pd.DataFrame(dict(zip(cols, data)))

    natural_positive_prob = sum(y_true)/float(len(y_true))

    df['bin'] = __pd.qcut(df['pred'], bins, duplicates='drop', labels=False)

    pos_group_df = df.groupby('bin')
    cnt_positive = pos_group_df['actual'].sum()
    cnt_all = pos_group_df['actual'].count()
    prob_avg = pos_group_df['pred'].mean()

    true_rate = pos_group_df['actual'].sum()/pos_group_df['actual'].count()
    lift = (true_rate/natural_positive_prob)

    cols = ['cnt_all', 'cnt_true', 'true_rate', 'pred_mean', 'lift', 'random_prob']
    data = [cnt_all, cnt_positive, true_rate, prob_avg, lift, natural_positive_prob]
    lift_df = __pd.DataFrame(dict(zip(cols, data)))

    return lift_df[cols]


def showlift(arr, bins=10):
    """
    Draws Calibration Plot

    Parameters
    -------------------
    arr: Array with data for multiple lines.
    [(pred1, true1, label1), (pred2, true2, label2)]

    bins: int, default=10
    Number of equal sized buckets to divide observations across
    """
    __plt.figure(figsize=(15,10))
    __plt.plot([1]*bins, c='k', ls='--', label='Random guess')
    for line in arr:
        pred = line[0]
        y_true = line[1]
        label = line[2]
        df_lift = get_lift_df(pred, y_true, bins)
        __plt.plot(df_lift.lift, label=label)

    __plt.xlabel("Bin", fontsize=20)
    __plt.xticks(range(bins), fontsize=15)
    __plt.ylabel("Lift", fontsize=20)
    __plt.yticks(fontsize=15)
    __plt.legend(fontsize=20)
    __plt.show()

#def del_from_list(A, B):
#    """
#	Removes list B from list A.
#    """
#    return [el for el in A if el not in B]

def showcm(cm, class_names=None):
    """
	Draws Confusion Matrix

    Parameters
    -------------------
    cm: Confusion Matrix. Can be obtained using confusion_matrix(y_true, y_pred) from sklearn.metrics
    """
    __plt.figure(figsize=(10,7.5))
    __plt.title('Confusion Matrix', fontsize=20)
    __sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", annot_kws={"size": 20}, linewidths=1)
    __plt.xlabel('Predicted', fontsize=20)
    __plt.ylabel('True', fontsize=20)

    if class_names is not None:
        tick_marks = __np.arange(len(class_names)) + 0.5
        __plt.xticks(tick_marks, class_names, fontsize=20)
        __plt.yticks(tick_marks, class_names, fontsize=20, rotation=0)
    else:
        __plt.xticks(fontsize=20)
        __plt.yticks(fontsize=20)

    __plt.show()

def _showcc(arr, NUM_SUBPLOT):
    df_arr = []
    for tr in arr:
        df_scores = __pd.DataFrame(list(zip(tr[0], tr[1])), columns=['score', 'target'])
        n_bins = len(__pd.qcut(df_scores.score,
                         10,
                         duplicates='drop',
                         retbins=True)[1]) - 1
        df_arr.append((df_scores.groupby([__pd.qcut(df_scores.score,
                                                 10,
                                                 range(n_bins),
                                                 duplicates='drop')]).agg(['mean'])[['score', 'target']], tr[2]))


    lim_down = 1
    lim_up = 0
    for df in df_arr:
        df = df[0]
        if min(df.score.min().iloc[0], df.target.min().iloc[0]) < lim_down:
            lim_down = min(df.score.min().iloc[0], df.target.min().iloc[0])

        if max(df.score.max().iloc[0], df.target.max().iloc[0]) > lim_up:
            lim_up = max(df.score.max().iloc[0], df.target.max().iloc[0])
    #__plt.figure(1)
    __plt.subplot(NUM_SUBPLOT)
    #__plt.figure(figsize=(10,10))
    __plt.title('Calibration Curves')
    __plt.plot([lim_down, lim_up], [lim_down, lim_up], "k:", label="Perfectly calibrated")

    for df in df_arr:
        __plt.plot(df[0].score, df[0].target, "s-",
             label=df[1])

    __plt.xlabel("Mean predicted value")

    __plt.xlim(lim_down - lim_down*.02, lim_up + lim_up*.02)
    __plt.ylim(lim_down - lim_down*.02, lim_up + lim_up*.02)

    __plt.xlim(lim_down - lim_down*.02, lim_up + lim_up*.02)
    __plt.ylim(lim_down - lim_down*.02, lim_up + lim_up*.02)

    __plt.ylabel("Mean real value")
    __plt.legend(loc='lower right')
    #__plt.show()

def showcc(arr, NUM_SUBPLOT=111):
    """
    Draws Calibration Curves

    Parameters
    -------------------
    arr: Array with data for multiple lines.
    [(pred1, true1, label1), (pred2, true2, label2)]
    """
    _showcc(arr, NUM_SUBPLOT)
    
def get_mte_dict(data, target, feature_names, alpha=5):
    """
    Params
    ---------
    data : pandas.Dataframe
        data without target
    target : pandas.Series
        target
    feature_names : str
        array with names of categorical features
    alpha : int / float
        regularization coefficient
    """
    d = {}
    
    for feature in feature_names:
        df_loc = __pd.DataFrame({'target': target,
                                 'feature': data[feature]})

        global_mean = __np.mean(target)
        nrows = df_loc['feature'].value_counts().sort_index()

        mean_target = df_loc.groupby(['feature'])['target'].mean()

        d[feature] = ((mean_target*nrows + global_mean*alpha)/(nrows+alpha)).to_dict()
    
    return d

def __lift1(y_test, y_pred, dcl):
    y_test, y_pred = __np.array(y_test), __np.array(y_pred)
    order = __np.argsort(y_pred)[::-1]
    y_pred = y_pred[order]
    y_test = y_test[order]
    
    fpr, tpr, thresholds = __roc_curve(y_test, y_pred, pos_label=1)
    auc_x = __auc(fpr, tpr)
    
    positive_rate = (float(sum(y_test)) / y_test.shape[0])
    df = __pd.DataFrame(columns=('k', 'lift_k', 'lift_cumulative', 'precision','tp','num_samples','threshold'))
    prev_total, prev_tp=0,0
    for i, percent in enumerate(dcl):
        num_samples = int(float(y_pred.shape[0])*percent/100)
        tp = sum( y_test[:num_samples] )
        total = y_pred[:num_samples].shape[0]+0.0000001
        #print num_samples, total, '\n'
        precision = float(tp)/total
        lift_cumulative = precision / positive_rate
        lift_k = (float(tp-prev_tp)/(total-prev_total)) / positive_rate
        threshold = y_pred[:num_samples][-1]
        df.loc[i] = [percent, lift_k, lift_cumulative, precision, tp, num_samples, threshold]
        
        #
        prev_total=total
        prev_tp=tp
    return df
def print_lift_roc_pr_plot(Y_TRUE, Y_PREDICTED, dcl = [1,5,10,20,30,40,50,60,70,80,90,100], CNT_TO_PRINT=10, NAME_MODEL='PTB', NFLAG_PRINT_TABLE = True, NUM_FIGURE=1, NUM_upper_text=0.05):

    lift_df_ptb = __lift1(Y_TRUE,Y_PREDICTED,dcl)
    if NFLAG_PRINT_TABLE:
        print (NAME_MODEL+ '\n' +str(lift_df_ptb[['k', 'lift_cumulative', 'precision', 'tp', 'num_samples','threshold']])+ '\n')
    
    s_lift_df_ptb = __lift1(Y_TRUE,Y_PREDICTED,[10,20,30,40,50,60,70,80,90,100])
    
    #__plt.figure(NUM_FIGURE)
    __plt.figure(NUM_FIGURE, figsize=(20,20))
    __plt.subplot(321)

    int_dcl = lift_df_ptb.k.astype(int)
    __plt.plot( int_dcl , lift_df_ptb['lift_cumulative'], label=NAME_MODEL,linewidth= 3 )
    __plt.xticks(int_dcl)
    #__plt.plot(int_dcl[0:CNT_TO_PRINT+1], lift_df_ptb['lift_cumulative'][0:CNT_TO_PRINT+1],'g^',color='r', label='Lift Value')
    __plt.plot(int_dcl[0:CNT_TO_PRINT+1], lift_df_ptb['lift_cumulative'][0:CNT_TO_PRINT+1],'o', label='Lift Value ' + NAME_MODEL)
    for i in zip(int_dcl[0:CNT_TO_PRINT+1], lift_df_ptb['lift_cumulative'][0:CNT_TO_PRINT+1]):
        __plt.text(i[0], i[1] + NUM_upper_text, str(round(i[1],2)), fontsize=11) # i[1] - 0.15

    __plt.legend(loc='upper right', fontsize=10, frameon=False)
    #plt.legend(frameon=False, fontsize=15)

    __plt.title("__LIFT_CUMULATIVE__") 
    __plt.ylabel('Lift')
    __plt.xlabel('Decile')
    
    __plt.subplot(322)
    int_dcl = s_lift_df_ptb.k.astype(int)
    __plt.plot( int_dcl , s_lift_df_ptb['lift_k'], label=NAME_MODEL,linewidth= 3 )
    __plt.xticks(int_dcl)
    __plt.plot(int_dcl[0:CNT_TO_PRINT+1], s_lift_df_ptb['lift_k'][0:CNT_TO_PRINT+1],'o', label='Lift Value ' + NAME_MODEL)
    for i in zip(int_dcl[0:CNT_TO_PRINT+1], s_lift_df_ptb['lift_k'][0:CNT_TO_PRINT+1]):
        __plt.text(i[0], i[1] + NUM_upper_text, str(round(i[1],2)), fontsize=11)

    __plt.legend(loc='upper right', fontsize=10, frameon=False)
    __plt.title("__LIFT__") 
    __plt.ylabel('Lift')
    __plt.xlabel('Decile')

    __plt.subplot(323)

    __plt.plot([0, 1], [0, 1], 'k--')

    fpr_ptb, tpr_ptb, _ = __roc_curve(Y_TRUE,Y_PREDICTED)

    __plt.plot(fpr_ptb, tpr_ptb
             , label=NAME_MODEL + ' ROC AUC={} GINI={}'.format( round(__roc_auc_score(Y_TRUE,Y_PREDICTED),3), str(round((__roc_auc_score(Y_TRUE,Y_PREDICTED)-0.5)*2.0,3)) )
             , linewidth= 2 ,linestyle='--')
    #plt.plot(fpr_ptb, tpr_ptb,linewidth= 2 ,linestyle='--')
    __plt.xlabel('False positive rate')
    __plt.ylabel('True positive rate')
    __plt.title('ROC', fontsize=15)
    __plt.legend(loc='best', fontsize=10)
    
    __plt.subplot(324)
    average_precision = __average_precision_score(Y_TRUE,Y_PREDICTED)

    #print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    precision_x, recall_x, _ = __precision_recall_curve(Y_TRUE,Y_PREDICTED)

    __plt.step(recall_x, precision_x, alpha=0.2,where='post', label=NAME_MODEL)
    __plt.fill_between(recall_x, precision_x, step='post', alpha=0.2)

    __plt.xlabel('Recall')
    __plt.ylabel('Precision')
    __plt.ylim([0.0, 1.05])
    __plt.xlim([0.0, 1.0])
    __plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(average_precision))
    __plt.legend(loc='best', fontsize=10)

    
    _showcc([(Y_PREDICTED, Y_TRUE, NAME_MODEL)],NUM_SUBPLOT=325)
    
def show_performance(X, y, clf, conf_matrix_cutoff=0.5, show_auc=True, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True):
    y_pred_proba=clf.predict_proba(X)[:,1]
    y_pred=[1 if i>=conf_matrix_cutoff else 0 for i in y_pred_proba]
    
    if show_auc:
        print ("AUC:{0:.3f}".format(__roc_auc_score(y,y_pred_proba)),"\n")
    if show_accuracy:
        print ("Accuracy:{0:.3f}".format(__accuracy_score(y,y_pred)),"\n")

    if show_classification_report:
        print ("Classification report")
        print (__classification_report(y,y_pred))
        
    if show_confusion_matrix:
        print ("Confusion matrix")
        showcm(__confusion_matrix(y, y_pred))
        print (__confusion_matrix(y,y_pred,labels=[0,1]),"\n")
        
        
        
def __print_GridSearchCV(results, name_param='min_samples_split'):
    __plt.figure(figsize=(13, 13))
    __plt.title("GridSearchCV evaluating using multiple scorers simultaneously",fontsize=16)

    __plt.xlabel(name_param)
    __plt.ylabel("Score")
    __plt.grid()

    ax = __plt.axes()
    #ax.set_xlim(0, 11)
    ax.set_ylim(0.5, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_'+name_param].data, dtype=float)

    for scorer, color in zip(sorted(__scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    __plt.legend(loc="best")
    __plt.grid('off')
    __plt.show()

def check_param_gs(__model, X, Y, par_to_check='min_samples_split', all_params = {'min_samples_split': 100}, p_range={}):
    if par_to_check in all_params:
        print ('CURR val: ', par_to_check, ': ', all_params[par_to_check])
        del all_params[par_to_check]
    gs = __GridSearchCV(__model.set_params(**all_params), # __xgb.XGBClassifier
                      param_grid={par_to_check:  p_range[par_to_check]},
                      scoring=__scoring, cv=5, refit='AUC', verbose=1)
    gs.fit(X, Y)
    results = gs.cv_results_
    all_params[par_to_check]=results['param_%s' % par_to_check][np.nonzero(results['rank_test_AUC'] == 1)[0][0]]
    print ('NEW val: ',par_to_check, ': ',all_params[par_to_check])
    __print_GridSearchCV(results,par_to_check)
    
    
def get_aucs(mdl = __LogisticRegression(n_jobs=10,C=1) , x_df = __pd.DataFrame(), Y_df = __pd.Series()):
    base_model = mdl
    X_Train_AUC=[]
    for idx, name in  __tqdm_notebook(enumerate(x_df.columns), ncols=x_df.shape[1]):
        skf = __StratifiedKFold(n_splits=5, random_state=999, shuffle=True)
        cv_score = __cross_val_score(base_model
                                      ,__StandardScaler().fit_transform(x_df[name].values.reshape(-1, 1))
                                      ,Y_df
                                      ,n_jobs=10,cv=skf,verbose=0,scoring='roc_auc')
        auc_avg, auc_std_95, auc_arr = __np.mean(cv_score), cv_score.std(), cv_score #np.std(cv_score)*2 
        X_Train_AUC.append([name, auc_avg, auc_std_95, auc_arr])
    i_ind=[item[0] for item in X_Train_AUC]
    X_Train_AUC=__pd.DataFrame(X_Train_AUC, columns=['NAME','AUC_AVG', 'AUC_STD_95', 'AUC_ARR'], index=i_ind)
    return X_Train_AUC
