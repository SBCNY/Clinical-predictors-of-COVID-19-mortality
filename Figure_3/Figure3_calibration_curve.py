from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import statsmodels.api as sm
import numpy as np
import pandas as pd

# predicted and actual score of both 3f and 17f model on test1 and test2 dataset
df1 = pd.read_csv('Predicted_&_actual_scores_3f_17F_test1.csv')
df2 = pd.read_csv('Predicted_&_actual_scores_3f_17F_test2.csv')

y_3f_pred_t1 = df1['score_3F_predicted']
y_17f_pred_t1 = df1['score_17F_predicted']
y_true_t1 = df1['test1_actual_label']

y_3f_pred_t2 = df2['score_3F_predicted']
y_17f_pred_t2 = df2['score_17F_predicted']
y_true_t2 = df2['test2_actual_label']

data_dict = {'Test set 1':[y_3f_pred_t1, y_17f_pred_t1, y_true_t1],
             'Test set 2':[y_3f_pred_t2, y_17f_pred_t2, y_true_t2]}

# create figure with 2x2 subplot
fig_size = (16,7)
fig1, ax = plt.subplots(2,2, figsize=(12,10))

fig1.tight_layout(pad=5.0)
fig1.subplots_adjust(wspace=0.45, hspace=0.25)
label_fs = 18
lw = 4
legend_size = 12

n_bootstraps = 1000
rng_seed = 0  # control reproducibility


rng = np.random.RandomState(rng_seed)

# bootstrapping AUC
def CI_auc(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    #    print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 95% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    # print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(confidence_lower, confidence_upper))
    return auc, confidence_lower, confidence_upper

# Estimate slope and intercept of calibration curve
def params_calibration_curve(y_true, y_pred):
    fop, mpv = calibration_curve(y_true, y_pred, n_bins=10)
    ols = sm.OLS(fop, sm.add_constant(mpv)).fit()
    intercept, slope = ols.params
    return intercept, slope

# # bootstrapping slope and intercept
def CI_calibration_curve(y_true, y_pred):
    intercept, slope = params_calibration_curve(y_true, y_pred)
    bootstrapped_intercept = []
    bootstrapped_slope = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for cablibration curve
            # to be defined: reject the sample
            continue

        intercept, slope = params_calibration_curve(y_true[indices], y_pred[indices])
        bootstrapped_intercept.append(intercept)
        bootstrapped_slope.append(slope)


    sorted_intercept = np.array(bootstrapped_intercept)
    sorted_intercept.sort()

    sorted_slope = np.array(bootstrapped_slope)
    sorted_slope.sort()

    # Computing the lower and upper bound of the 95% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower_intercept = sorted_intercept[int(0.025 * len(sorted_intercept))]
    confidence_upper_intercept = sorted_intercept[int(0.975 * len(sorted_intercept))]

    confidence_lower_slope = sorted_slope[int(0.025 * len(sorted_slope))]
    confidence_upper_slope = sorted_slope[int(0.975 * len(sorted_slope))]
    return {'slope': [slope, confidence_lower_slope, confidence_upper_slope],
            'intercept': [intercept, confidence_lower_intercept, confidence_upper_intercept]}



for idx, (k, v) in enumerate(data_dict.items()):

    ax1 = ax[0, idx]
    ax2 = ax[1, idx]

    # plot AUC curve
    fpr_3f, tpr_3f, _ = roc_curve(v[-1], v[0])
    auc_3f, auc_3f_l, auc_3f_u = CI_auc(v[-1], v[0])

    fpr_17f, tpr_17f, _ = roc_curve(v[-1], v[1])
    auc_17f, auc_17f_l , auc_17f_u = CI_auc(v[-1], v[1])

    ax1.plot(fpr_17f, tpr_17f, linewidth=lw,color='darkorange', label='17F Model\n(AUC = {:.2f} [{:.2f}-{:.2f}])'.format(auc_17f, auc_17f_l, auc_17f_u))
    ax1.plot(fpr_3f, tpr_3f, linewidth=lw, color='blue', label='3F Model\n(AUC = {:.2f} [{:.2f}-{:.2f}])'.format(auc_3f, auc_3f_l, auc_3f_u))

    ax1.set_xlabel('False Positive Rate', fontsize=label_fs)
    ax1.set_ylabel('True Positive Rate', fontsize=label_fs)
    ax1.legend(fontsize=legend_size, prop=dict(weight='bold'))
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.plot([0, 1], [0, 1], lw=lw, color='black', linestyle='--')
    if idx == 0:
        ax1.text(-0.45, 0.5, '(A)', fontsize=28, weight='bold')
    else:
        ax1.text(-0.45, 0.5, '(B)', fontsize=28, weight='bold')
    ax1.set_title(k, fontsize=20)

    # plot calibration curve
    fop_3f, mpv_3f = calibration_curve(v[-1], v[0], n_bins=10)
    c3f = CI_calibration_curve(v[-1], v[0])
    c3f_s = c3f['slope']
    c3f_i = c3f['intercept']


    fop_17f, mpv_17f = calibration_curve(v[-1], v[1], n_bins=10)
    c17f = CI_calibration_curve(v[-1], v[1])
    c17f_s = c17f['slope']
    c17f_i = c17f['intercept']
    ax2.plot(mpv_17f, fop_17f, "s-", color='darkorange', linewidth=lw,
             label='17F Model\n(S = {:.2f} [{:.2f}-{:.2f}],\nI = {:.2f} [{:.2f}-{:.2f}])'.format(c17f_s[0], c17f_s[1], c17f_s[2],
                                                                                                          c17f_i[0], c17f_i[1], c17f_i[2]))
    ax2.plot(mpv_3f, fop_3f, "s-", color='blue', linewidth=lw,
             label='3F Model\n(S = {:.2f} [{:.2f}-{:.2f}],\nI = {:.2f} [{:.2f}-{:.2f}])'.format(c3f_s[0], c3f_s[1], c3f_s[2],
                                                                                                   c3f_i[0], c3f_i[1], c3f_i[2]))
    ax2.set_xlabel('Mean Predicted Value', fontsize=label_fs)
    ax2.set_ylabel('Fraction of Positives', fontsize=label_fs)
    ax2.legend(loc=4, fontsize=legend_size, prop=dict(weight='bold'))
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    if idx == 0:
        ax2.text(-0.45, 0.5, '(C)', fontsize=28, weight='bold')
    else:
        ax2.text(-0.45, 0.5, '(D)', fontsize=28, weight='bold')

# save figure
fig1.savefig('figure3.tif', bbox_inches="tight", dpi=600, pil_kwargs={"compression": "tiff_lzw"})


