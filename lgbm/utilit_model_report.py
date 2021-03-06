#import cx_Oracle as __orcl
import datetime as __datetime
import pandas as __pd
import numpy as __np
import matplotlib.pyplot as __plt
import seaborn as __sns
import scikitplot as __skplt
import itertools as __itertools 
from tqdm import tqdm_notebook as __tqdm_notebook
from sklearn.preprocessing import StandardScaler as __StandardScaler
from matplotlib.gridspec import GridSpec as __GridSpec
from scikitplot.helpers import binary_ks_curve as __binary_ks_curve
from scikitplot.helpers import cumulative_gain_curve as __cumulative_gain_curve
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
                         retbins=True,
                         precision=5)[1]) - 1
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
    __plt.legend(loc='lower right', fontsize=11)
    #__plt.show()
    
def _show_acclift(arr, NUM_SUBPLOT=111, dcl=[1,5,10,20,30,40,50,60,70,80,90,100], CNT_TO_PRINT=10):
    colors = __itertools.cycle(["royalblue", "darkorange", "forestgreen"])
    colors2 = __itertools.cycle(["royalblue", "darkorange", "forestgreen"])
    for tr in arr:
        df_scores = __pd.DataFrame(list(zip(tr[0], tr[1])), columns=['score', 'target'])
        lift_df_ptb = __lift1(df_scores['target'], df_scores['score'], dcl)
        int_dcl = lift_df_ptb.k.astype(int)
        __plt.subplot(NUM_SUBPLOT)
        __plt.plot(int_dcl , lift_df_ptb['lift_cumulative'], color=next(colors), label=tr[2], linewidth=2, alpha=0.7)
        __plt.plot(int_dcl[0:CNT_TO_PRINT+1], lift_df_ptb['lift_cumulative'][0:CNT_TO_PRINT+1],'o', color=next(colors2), label=tr[2])
        for i in zip(int_dcl[0:CNT_TO_PRINT+1], lift_df_ptb['lift_cumulative'][0:CNT_TO_PRINT+1]):
            __plt.text(i[0], i[1], str(round(i[1],2)), horizontalalignment='center', verticalalignment='bottom', fontsize=13)
        
        __plt.legend(loc='upper right', fontsize=12, frameon=False)
        __plt.title("LIFT CUMULATIVE", fontsize=13) 
        __plt.ylabel('Lift')
        __plt.xlabel('Decile')
        
def _showlift(arr, NUM_SUBPLOT=111, dcl = [10,20,30,40,50,60,70,80,90,100], CNT_TO_PRINT=10):
    colors = __itertools.cycle(["royalblue", "darkorange", "forestgreen"])
    colors2 = __itertools.cycle(["royalblue", "darkorange", "forestgreen"])
    for tr in arr:
        df_scores = __pd.DataFrame(list(zip(tr[0], tr[1])), columns=['score', 'target'])
        s_lift_df_ptb = __lift1(df_scores['target'], df_scores['score'], dcl)
        int_dcl = s_lift_df_ptb.k.astype(int)
        __plt.plot(int_dcl, s_lift_df_ptb['lift_k'], label=tr[2], linewidth=2, color=next(colors), alpha=0.7)
        __plt.xticks(int_dcl)
        __plt.plot(int_dcl[0:CNT_TO_PRINT+1], s_lift_df_ptb['lift_k'][0:CNT_TO_PRINT+1],'o', label=tr[2], color=next(colors2))
        for i in zip(int_dcl[0:CNT_TO_PRINT+1], s_lift_df_ptb['lift_k'][0:CNT_TO_PRINT+1]):
            __plt.text(i[0], i[1], str(round(i[1],2)), horizontalalignment='center', verticalalignment='bottom', fontsize=13)
        
        __plt.legend(loc='upper right', fontsize=12, frameon=False)
        __plt.title("LIFT", fontsize=13) 
        __plt.ylabel('Lift')
        __plt.xlabel('Decile')
        
def _showroccurve(arr, NUM_SUBPLOT=111):
    colors = __itertools.cycle(["royalblue", "darkorange", "forestgreen"])
    for tr in arr:
        df_scores = __pd.DataFrame(list(zip(tr[0], tr[1])), columns=['score', 'target'])
        fpr_ptb, tpr_ptb, _ = __roc_curve(df_scores['target'], df_scores['score'])
        __plt.plot([0, 1], [0, 1], 'k--')
        __plt.plot(fpr_ptb, tpr_ptb, label=str(tr[2])+' ROC AUC={} GINI={}'.format( round(__roc_auc_score(df_scores['target'], df_scores['score']),3), str(round((__roc_auc_score(df_scores['target'], df_scores['score'])-0.5)*2.0,3)) ), linewidth= 2 ,linestyle='--', color=next(colors), alpha=0.7)
        __plt.xlabel('False positive rate')
        __plt.ylabel('True positive rate')
        __plt.title('ROC', fontsize=13)
        __plt.legend(loc='best', fontsize=12)
        
def _gainchart(arr, NUM_SUBPLOT=111):
    colors = __itertools.cycle(["royalblue", "darkorange", "forestgreen"])
    colors2 = __itertools.cycle(["royalblue", "darkorange", "forestgreen"])
    for tr in arr:
        df_scores = __pd.DataFrame(list(zip(tr[0], tr[1])), columns=['score', 'target'])
        y_pred = __pd.DataFrame(columns=['pred_0', 'pred_1'], dtype=float)
        y_pred['pred_0'] = 1-df_scores['score']
        y_pred['pred_1'] = df_scores['score']
        y_true = __np.array(df_scores['target'])
        y_probas = __np.array(y_pred)
        classes = __np.unique(y_true)
        percentages, gains2 = __cumulative_gain_curve(y_true, y_probas[:, 1], classes[1])
        decile = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        indexes = []
        
        for i in list(percentages):
            if round(i,4) in decile:
                indexes.append(list(percentages).index(i))
                
        __plt.plot(percentages[indexes], gains2[indexes], lw=2, label=tr[2], color=next(colors), alpha=0.7)
        __plt.plot(percentages[indexes], gains2[indexes], 'o', label=tr[2], color=next(colors2), alpha=0.7)
        
        for i in zip(percentages[indexes], gains2[indexes]):
            __plt.text(i[0], i[1], str(round(i[1],2)),  horizontalalignment='center', verticalalignment='bottom',fontsize=13)
        
        __plt.xticks(decile)
        __plt.ylim([0.0, 1.05])
        
        __plt.plot([0, 1], [0, 1], 'k--', lw=2)
        __plt.xlabel('Percentage of sample')
        __plt.ylabel('Gain')
        __plt.tick_params()
        __plt.title('Gain chart', fontsize=13)
        __plt.legend(loc='lower right', fontsize=12)
        
def _kschart(arr, NUM_SUBPLOT=111):
    colors = __itertools.cycle(["royalblue", "darkorange", "forestgreen"])
    colors2 = __itertools.cycle(["royalblue", "darkorange", "forestgreen"])
    colors3 = __itertools.cycle(["royalblue", "darkorange", "forestgreen"])
    for tr in arr:
        df_scores = __pd.DataFrame(list(zip(tr[0], tr[1])), columns=['score', 'target'])
        thresholds, pct1, pct2, ks_statistic, max_distance_at, classes = __binary_ks_curve(df_scores['target'], df_scores['score'].ravel())
        __plt.plot(thresholds, pct1, lw=2, color=next(colors), alpha=0.7, label=str(tr[2]) + ' class {}'.format(classes[0]))
        __plt.plot(thresholds, pct2, lw=2, color=next(colors2), alpha=0.7, label=str(tr[2]) + ' class {}'.format(classes[1]))
        idx = __np.where(thresholds == max_distance_at)[0][0]
        __plt.axvline(max_distance_at, *sorted([pct1[idx], pct2[idx]]),
                  label= str(tr[2])+' KS Statistic: {:.3f} at {:.3f}'.format(ks_statistic, max_distance_at),
                  linestyle=':', lw=2, color=next(colors3))
        
        __plt.xlim([0.0, 1.01])
        __plt.ylim([0.0, 1.01])
        __plt.xlabel('Threshold')
        __plt.ylabel('Percentage below threshold')
        __plt.title('K-S test', fontsize=13)
        __plt.legend(loc='lower right', fontsize=11)
            
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


def print_report_ttv(Y_TRUE_TRAIN, Y_PREDICTED_TRAIN, Y_TRUE_TEST, Y_PREDICTED_TEST, Y_TRUE_VALID, Y_PREDICTED_VALID,
                 FEAT_IMP='', dcl = [1,5,10,20,30,40,50,60,70,80,90,100], CNT_TO_PRINT=10, NAME_MODEL='PTB', NFLAG_PRINT_TABLES = True,
                 TARGET_NAME='', DATA_SET='', METHOD='', TIME_PERIOD='', UNDER='', CALIBR='', PYTHON_SCRIPT='', ORACLE_TABLE='',
                 CSV_FILE='', SQL_FILE=''):
    
    gs = __GridSpec(9, 1, left=0, right=0.2, hspace=0.3, wspace=0.3)

    __plt.figure(figsize=(55,75))
    __plt.subplot(gs[0,0])

    #int_dcl = lift_df_ptb.k.astype(int)
    
    
    _show_acclift([(Y_PREDICTED_TRAIN, Y_TRUE_TRAIN, 'Train'), (Y_PREDICTED_TEST, Y_TRUE_TEST, 'Test'), (Y_PREDICTED_VALID, Y_TRUE_VALID, 'Valid')], NUM_SUBPLOT=gs[0, 0])
    
    __plt.subplot(gs[1, 0])
    
    _showlift([(Y_PREDICTED_TRAIN, Y_TRUE_TRAIN, 'Train'), (Y_PREDICTED_TEST, Y_TRUE_TEST, 'Test'), (Y_PREDICTED_VALID, Y_TRUE_VALID, 'Valid')], NUM_SUBPLOT=gs[1, 0])
    
    __plt.subplot(gs[2, 0])

    _showroccurve([(Y_PREDICTED_TRAIN, Y_TRUE_TRAIN, 'Train'), (Y_PREDICTED_TEST, Y_TRUE_TEST, 'Test'), (Y_PREDICTED_VALID, Y_TRUE_VALID, 'Valid')], NUM_SUBPLOT=gs[2, 0])
    
    __plt.subplot(gs[3, 0])
    
    average_precision = __average_precision_score(Y_TRUE_TRAIN,Y_PREDICTED_TRAIN)

    precision_x, recall_x, _ = __precision_recall_curve(Y_TRUE_TRAIN,Y_PREDICTED_TRAIN)

    __plt.step(recall_x, precision_x, alpha=0.2,where='post', label=NAME_MODEL)
    __plt.fill_between(recall_x, precision_x, step='post', alpha=0.2)

    __plt.xlabel('Recall')
    __plt.ylabel('Precision')
    __plt.ylim([0.0, 1.05])
    __plt.xlim([0.0, 1.0])
    __plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(average_precision))
    __plt.legend(loc='best', fontsize=11)
    
    __plt.subplot(gs[4, 0])
    
    width = 0.35
    ind = __np.arange(10) 
    x_test=get_lift_df(Y_PREDICTED_TEST,Y_TRUE_TEST)['lift']
    x_valid=get_lift_df(Y_PREDICTED_VALID,Y_TRUE_VALID)['lift']
    __plt.bar(ind, x_test, width, label='test', color='orange', alpha = 0.6)
    __plt.bar(ind+width, x_valid, width, label='valid', color='green', alpha = 0.6)
    __plt.xlabel('Part')
    __plt.ylabel('Lift')
    __plt.xticks(ind + width / 2, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    __plt.title('DECILE_LIFT', fontsize=23)
    __plt.title(r'Decile lift')
    for i in zip(ind, x_test):
        __plt.text(i[0], i[1], str(round(i[1],2)), horizontalalignment='center', verticalalignment='bottom', fontsize=13)
        
    for j in zip(ind+width, x_valid):
        __plt.text(j[0], j[1], str(round(j[1],2)), horizontalalignment='center', verticalalignment='bottom', fontsize=13)
    
    __plt.legend(loc='best', fontsize=15)
    
    __plt.subplot(gs[5, 0])
    
    width = 0.35
    ind = __np.arange(10) 
    z_test=get_lift_df(Y_PREDICTED_TEST,Y_TRUE_TEST)['pred_mean']
    z_valid=get_lift_df(Y_PREDICTED_VALID,Y_TRUE_VALID)['pred_mean']
    __plt.bar(ind, z_test, width, label='test', color='orange', alpha = 0.6)
    __plt.bar(ind+width, z_valid, width, label='valid', color='green', alpha = 0.6)
    __plt.xlabel('Part')
    __plt.ylabel('Responce')
    __plt.xticks(ind + width / 2, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    __plt.title('DECILE_RESPONCE', fontsize=23)
    __plt.title(r'Decile responce')
    for i in zip(ind, z_test):
        __plt.text(i[0], i[1], str(round(i[1],2)), horizontalalignment='center', verticalalignment='bottom', fontsize=13)
        
    for j in zip(ind+width, z_valid):
        __plt.text(j[0], j[1], str(round(j[1],2)), horizontalalignment='center', verticalalignment='bottom', fontsize=13)    
    
    __plt.legend(loc='best', fontsize=15)
    
    __plt.subplot(gs[6, 0])
    
    _gainchart([(Y_PREDICTED_TRAIN, Y_TRUE_TRAIN, 'Train'), (Y_PREDICTED_TEST, Y_TRUE_TEST, 'Test'), (Y_PREDICTED_VALID, Y_TRUE_VALID, 'Valid')], NUM_SUBPLOT=gs[6, 0])
    
    __plt.subplot(gs[7, 0])
    
    _kschart([(Y_PREDICTED_TRAIN, Y_TRUE_TRAIN, 'Train'), (Y_PREDICTED_TEST, Y_TRUE_TEST, 'Test'), (Y_PREDICTED_VALID, Y_TRUE_VALID, 'Valid')], NUM_SUBPLOT=gs[7, 0])
    
    __plt.subplot(gs[8, 0])
    
    _showcc([(Y_PREDICTED_TRAIN, Y_TRUE_TRAIN, 'Train'), (Y_PREDICTED_TEST, Y_TRUE_TEST, 'Test'), (Y_PREDICTED_VALID, Y_TRUE_VALID, 'Valid')], NUM_SUBPLOT=gs[8, 0])
    
    
    info = __pd.DataFrame(columns=['BASE', 'TARGET', 'TARGET, %', 'ROC_AUC', 'GINI', 'Target name', 'Data set name','Model method',
                                       'Time period','Undersampling usage', 'Calibrated', 'Python script location',
                                       'Data set oracle location', 'Data set csv location', 'SQL code location'], index=range(1))
    info.loc[0,'BASE'] = len(Y_TRUE_TRAIN)
    info.loc[0,'TARGET'] = int(sum(Y_TRUE_TRAIN))
    info.loc[0,'TARGET, %'] = float(sum(Y_TRUE_TRAIN)/len(Y_TRUE_TRAIN))*100
    info.loc[0,'ROC_AUC'] = __roc_auc_score(Y_TRUE_TRAIN,Y_PREDICTED_TRAIN)
    info.loc[0,'GINI'] = 2*__roc_auc_score(Y_TRUE_TRAIN,Y_PREDICTED_TRAIN) - 1
    
    info.loc[1,'BASE'] = len(Y_TRUE_TEST)
    info.loc[1,'TARGET'] = int(sum(Y_TRUE_TEST))
    info.loc[1,'TARGET, %'] = float(sum(Y_TRUE_TEST)/len(Y_TRUE_TEST))*100
    info.loc[1,'ROC_AUC'] = __roc_auc_score(Y_TRUE_TEST,Y_PREDICTED_TEST)
    info.loc[1,'GINI'] = 2*__roc_auc_score(Y_TRUE_TEST,Y_PREDICTED_TEST) - 1
    
    info.loc[2,'BASE'] = len(Y_TRUE_VALID)
    info.loc[2,'TARGET'] = int(sum(Y_TRUE_VALID))
    info.loc[2,'TARGET, %'] = float(sum(Y_TRUE_VALID)/len(Y_TRUE_VALID))*100
    info.loc[2,'ROC_AUC'] = __roc_auc_score(Y_TRUE_VALID,Y_PREDICTED_VALID)
    info.loc[2,'GINI'] = 2*__roc_auc_score(Y_TRUE_VALID,Y_PREDICTED_VALID) - 1
   
    info.loc[0,'Target name'] = TARGET_NAME
    info.loc[0,'Data set name'] = DATA_SET
    info.loc[0,'Model method'] = METHOD
    info.loc[0,'Time period'] = TIME_PERIOD
    info.loc[0,'Undersampling usage'] = UNDER
    info.loc[0,'Calibrated'] = CALIBR
    info.loc[0,'Python script location'] = PYTHON_SCRIPT
    info.loc[0,'Data set oracle location'] = ORACLE_TABLE
    info.loc[0,'Data set csv location'] = CSV_FILE
    info.loc[0,'SQL code location'] = SQL_FILE
    
    info_t = info.T
    info_t.rename(columns={0: 'Train', 1: 'Test', 2: 'Valid'}, inplace=True)
    
    if NFLAG_PRINT_TABLES==True:
        print ('Model info')
        print (__pd.DataFrame(info_t))
        print ('________________________________________________________')
        print ('Feature importance')
        print (__pd.DataFrame(FEAT_IMP))
        print ('________________________________________________________')
    
    #if NFLAG_PRINT_IMP==True:
    #    return (pd.DataFrame(FEAT_IMP))
    
def print_report_t(Y_TRUE_TRAIN, Y_PREDICTED_TRAIN, FEAT_IMP='', dcl = [1,5,10,20,30,40,50,60,70,80,90,100], CNT_TO_PRINT=10, NAME_MODEL='PTB', NFLAG_PRINT_TABLES = True,TARGET_NAME='', DATA_SET='', METHOD='', TIME_PERIOD='', UNDER='', CALIBR='', PYTHON_SCRIPT='', ORACLE_TABLE='',CSV_FILE='', SQL_FILE=''):
    
    gs = __GridSpec(9, 1, left=0, right=0.2, hspace=0.3, wspace=0.3)

    __plt.figure(figsize=(55,75))
    __plt.subplot(gs[0,0])

    #int_dcl = lift_df_ptb.k.astype(int)
    
    
    _show_acclift([(Y_PREDICTED_TRAIN, Y_TRUE_TRAIN, 'Train')], NUM_SUBPLOT=gs[0, 0])
    
    __plt.subplot(gs[1, 0])
    
    _showlift([(Y_PREDICTED_TRAIN, Y_TRUE_TRAIN, 'Train')], NUM_SUBPLOT=gs[1, 0])
    
    __plt.subplot(gs[2, 0])

    _showroccurve([(Y_PREDICTED_TRAIN, Y_TRUE_TRAIN, 'Train')], NUM_SUBPLOT=gs[2, 0])
    
    __plt.subplot(gs[3, 0])
    
    average_precision = __average_precision_score(Y_TRUE_TRAIN,Y_PREDICTED_TRAIN)

    precision_x, recall_x, _ = __precision_recall_curve(Y_TRUE_TRAIN,Y_PREDICTED_TRAIN)

    __plt.step(recall_x, precision_x, alpha=0.2,where='post', label=NAME_MODEL)
    __plt.fill_between(recall_x, precision_x, step='post', alpha=0.2)

    __plt.xlabel('Recall')
    __plt.ylabel('Precision')
    __plt.ylim([0.0, 1.05])
    __plt.xlim([0.0, 1.0])
    __plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(average_precision))
    __plt.legend(loc='best', fontsize=11)
    
    __plt.subplot(gs[4, 0])
    
    _gainchart([(Y_PREDICTED_TRAIN, Y_TRUE_TRAIN, 'Train')], NUM_SUBPLOT=gs[4, 0])
    
    __plt.subplot(gs[5, 0])
    
    _kschart([(Y_PREDICTED_TRAIN, Y_TRUE_TRAIN, 'Train')], NUM_SUBPLOT=gs[5, 0])
    
    __plt.subplot(gs[6, 0])
    
    _showcc([(Y_PREDICTED_TRAIN, Y_TRUE_TRAIN, 'Train')], NUM_SUBPLOT=gs[6, 0])
    
    
    info = __pd.DataFrame(columns=['BASE', 'TARGET', 'TARGET, %', 'ROC_AUC', 'GINI', 'Target name', 'Data set name','Model method',
                                       'Time period','Undersampling usage', 'Calibrated', 'Python script location',
                                       'Data set oracle location', 'Data set csv location', 'SQL code location'], index=range(1))
    info.loc[0,'BASE'] = len(Y_TRUE_TRAIN)
    info.loc[0,'TARGET'] = int(sum(Y_TRUE_TRAIN))
    info.loc[0,'TARGET, %'] = float(sum(Y_TRUE_TRAIN)/len(Y_TRUE_TRAIN))*100
    info.loc[0,'ROC_AUC'] = __roc_auc_score(Y_TRUE_TRAIN,Y_PREDICTED_TRAIN)
    info.loc[0,'GINI'] = 2*__roc_auc_score(Y_TRUE_TRAIN,Y_PREDICTED_TRAIN) - 1
   
    info.loc[0,'Target name'] = TARGET_NAME
    info.loc[0,'Data set name'] = DATA_SET
    info.loc[0,'Model method'] = METHOD
    info.loc[0,'Time period'] = TIME_PERIOD
    info.loc[0,'Undersampling usage'] = UNDER
    info.loc[0,'Calibrated'] = CALIBR
    info.loc[0,'Python script location'] = PYTHON_SCRIPT
    info.loc[0,'Data set oracle location'] = ORACLE_TABLE
    info.loc[0,'Data set csv location'] = CSV_FILE
    info.loc[0,'SQL code location'] = SQL_FILE
    
    info_t = info.T
    info_t.rename(columns={0: 'Train'}, inplace=True)
    
    if NFLAG_PRINT_TABLES==True:
        print ('Model info')
        print (__pd.DataFrame(info_t))
        print ('________________________________________________________')
        print ('Feature importance')
        print (__pd.DataFrame(FEAT_IMP))
        print ('________________________________________________________')

def print_report_tt(Y_TRUE_TRAIN, Y_PREDICTED_TRAIN, Y_TRUE_TEST, Y_PREDICTED_TEST, FEAT_IMP='', dcl =[1,5,10,20,30,40,50,60,70,80,90,100], CNT_TO_PRINT=10, NAME_MODEL='PTB', NFLAG_PRINT_TABLES = True, TARGET_NAME='', DATA_SET='', METHOD='', TIME_PERIOD='', UNDER='', CALIBR='', PYTHON_SCRIPT='', ORACLE_TABLE='', CSV_FILE='', SQL_FILE=''):
    
    gs = __GridSpec(9, 1, left=0, right=0.2, hspace=0.3, wspace=0.3)

    __plt.figure(figsize=(55,75))
    __plt.subplot(gs[0,0])

    _show_acclift([(Y_PREDICTED_TRAIN, Y_TRUE_TRAIN, 'Train'), (Y_PREDICTED_TEST, Y_TRUE_TEST, 'Test')], NUM_SUBPLOT=gs[0, 0])
    
    __plt.subplot(gs[1, 0])
    
    _showlift([(Y_PREDICTED_TRAIN, Y_TRUE_TRAIN, 'Train'), (Y_PREDICTED_TEST, Y_TRUE_TEST, 'Test')], NUM_SUBPLOT=gs[1, 0])
    
    __plt.subplot(gs[2, 0])

    _showroccurve([(Y_PREDICTED_TRAIN, Y_TRUE_TRAIN, 'Train'), (Y_PREDICTED_TEST, Y_TRUE_TEST, 'Test')], NUM_SUBPLOT=gs[2, 0])
    
    __plt.subplot(gs[3, 0])
    
    width = 0.35
    ind = __np.arange(10) 
    x_test=get_lift_df(Y_PREDICTED_TEST,Y_TRUE_TEST)['lift']
    __plt.bar(ind, x_test, width, label='test', color='orange', alpha = 0.6)
    __plt.xlabel('Part')
    __plt.ylabel('Lift')
    __plt.xticks(ind + width / 2, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    __plt.title('DECILE_LIFT', fontsize=23)
    __plt.title(r'Decile lift')
    for i in zip(ind, x_test):
        __plt.text(i[0], i[1], str(round(i[1],2)), horizontalalignment='center', verticalalignment='bottom', fontsize=13)
    
    __plt.legend(loc='best', fontsize=15)
    
    __plt.subplot(gs[4, 0])
    
    width = 0.35
    ind = __np.arange(10) 
    z_test=get_lift_df(Y_PREDICTED_TEST,Y_TRUE_TEST)['pred_mean']
    __plt.bar(ind, z_test, width, label='test', color='orange', alpha = 0.6)
    __plt.xlabel('Part')
    __plt.ylabel('Responce')
    __plt.xticks(ind + width / 2, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    __plt.title('DECILE_RESPONCE', fontsize=23)
    __plt.title(r'Decile responce')
    for i in zip(ind, z_test):
        __plt.text(i[0], i[1], str(round(i[1],2)), horizontalalignment='center', verticalalignment='bottom', fontsize=13)
    
    __plt.legend(loc='best', fontsize=15)
    
    __plt.subplot(gs[5, 0])
    
    _gainchart([(Y_PREDICTED_TRAIN, Y_TRUE_TRAIN, 'Train'), (Y_PREDICTED_TEST, Y_TRUE_TEST, 'Test')], NUM_SUBPLOT=gs[5, 0])
    
    __plt.subplot(gs[6, 0])
    
    _kschart([(Y_PREDICTED_TRAIN, Y_TRUE_TRAIN, 'Train'), (Y_PREDICTED_TEST, Y_TRUE_TEST, 'Test')], NUM_SUBPLOT=gs[6, 0])
    
    __plt.subplot(gs[7, 0])
    
    _showcc([(Y_PREDICTED_TRAIN, Y_TRUE_TRAIN, 'Train'), (Y_PREDICTED_TEST, Y_TRUE_TEST, 'Test')], NUM_SUBPLOT=gs[7, 0])
    
    
    info = __pd.DataFrame(columns=['BASE', 'TARGET', 'TARGET, %', 'ROC_AUC', 'GINI', 'Target name', 'Data set name','Model method',
                                       'Time period','Undersampling usage', 'Calibrated', 'Python script location',
                                       'Data set oracle location', 'Data set csv location', 'SQL code location'], index=range(1))
    info.loc[0,'BASE'] = len(Y_TRUE_TRAIN)
    info.loc[0,'TARGET'] = int(sum(Y_TRUE_TRAIN))
    info.loc[0,'TARGET, %'] = float(sum(Y_TRUE_TRAIN)/len(Y_TRUE_TRAIN))*100
    info.loc[0,'ROC_AUC'] = __roc_auc_score(Y_TRUE_TRAIN,Y_PREDICTED_TRAIN)
    info.loc[0,'GINI'] = 2*__roc_auc_score(Y_TRUE_TRAIN,Y_PREDICTED_TRAIN) - 1
    
    info.loc[1,'BASE'] = len(Y_TRUE_TEST)
    info.loc[1,'TARGET'] = int(sum(Y_TRUE_TEST))
    info.loc[1,'TARGET, %'] = float(sum(Y_TRUE_TEST)/len(Y_TRUE_TEST))*100
    info.loc[1,'ROC_AUC'] = __roc_auc_score(Y_TRUE_TEST,Y_PREDICTED_TEST)
    info.loc[1,'GINI'] = 2*__roc_auc_score(Y_TRUE_TEST,Y_PREDICTED_TEST) - 1

    info.loc[0,'Target name'] = TARGET_NAME
    info.loc[0,'Data set name'] = DATA_SET
    info.loc[0,'Model method'] = METHOD
    info.loc[0,'Time period'] = TIME_PERIOD
    info.loc[0,'Undersampling usage'] = UNDER
    info.loc[0,'Calibrated'] = CALIBR
    info.loc[0,'Python script location'] = PYTHON_SCRIPT
    info.loc[0,'Data set oracle location'] = ORACLE_TABLE
    info.loc[0,'Data set csv location'] = CSV_FILE
    info.loc[0,'SQL code location'] = SQL_FILE
    
    info_t = info.T
    info_t.rename(columns={0: 'Train', 1: 'Test'}, inplace=True)
    
    if NFLAG_PRINT_TABLES==True:
        print ('Model info')
        print (__pd.DataFrame(info_t))
        print ('________________________________________________________')
        print ('Feature importance')
        print (__pd.DataFrame(FEAT_IMP))
        print ('________________________________________________________')    
    
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
        