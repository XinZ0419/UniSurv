import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def KMLinearExt(timeline, sur_prob, how):
    slopeLiner = (1 - sur_prob[-1]) / (0 - timeline[-1])
    KMLinearZero = -1 / slopeLiner
    timelineExt = np.linspace(start=timeline[-1], stop=KMLinearZero,
                              num=round((KMLinearZero - timeline[-1]) / np.mean(np.diff(timeline))))
    sur_probExt = np.maximum(1 + timelineExt * slopeLiner, 0)
    new_timeline = np.concatenate((timeline, timelineExt[1:]))
    new_surprob = np.concatenate((sur_prob, sur_probExt[1:]))
    new_timeSlot = np.diff(new_timeline)
    assert len(new_timeline) == len(timeline) + len(timelineExt) - 1
    assert len(new_surprob) == len(sur_prob) + len(sur_probExt) - 1
    if how == 'toTmax':
        return timeline, sur_prob, np.diff(timeline)
    elif how == 'Linearext':
        return new_timeline, new_surprob, new_timeSlot


def mae_margin_info(duration, event_indicator, how='toTmax'):
    assert len(duration) == len(event_indicator)
    kmf = KaplanMeierFitter()
    kmf.fit(durations=duration, event_observed=event_indicator)
    km_estimate = kmf.survival_function_  # km_estimate is a dataframe: index is timeline, first col is Survival curve

    # get the raw numerator array ready
    timeline, sur_prob = km_estimate.index.values, km_estimate['KM_estimate'].values

    timeline, sur_prob, time_slot = KMLinearExt(timeline, sur_prob, how)
    TmaxLinear = timeline[-1]

    sur_prob_slot = ((np.roll(sur_prob, -1) + sur_prob) / 2)[:-1]
    lifetime_slot = time_slot * sur_prob_slot
    lifetime_flipcum = np.pad(np.flip(np.cumsum(np.flip(lifetime_slot))), (0, 1))  # add one 0 at the end (this could be updated by extrapolation)
    assert len(lifetime_flipcum) == len(timeline)

    S_ck = np.array([(float(sur_prob[dur == timeline])) for dur in duration])
    alpha_k = event_indicator + (1 - event_indicator) * (1 - S_ck)
    cum_S_ck = np.array([(float(lifetime_flipcum[dur == timeline])) for dur in duration])
    mae_margin = duration + (1 - event_indicator) * cum_S_ck / S_ck
    assert len(alpha_k) == len(event_indicator)
    # print(type(S_ck))
    plt.plot(timeline, sur_prob, linestyle=':')
    plt.show()
    return mae_margin.astype(int), alpha_k, TmaxLinear


if __name__ == '__main__':
    # ms = pd.read_csv('../data/cro_val_0/train_labels_0.csv')
    ms = pd.read_csv('../../data 3/data_final_accu_cp_model3.csv')
    duration = ms['stop'] // 30
    event_indicator = ms['event_accu']
    mae_margin, alpha_k, Tmax = mae_margin_info(duration, event_indicator, how='toTmax')
    # print(mae_margin[event_indicator == 1])
    # print(duration[event_indicator == 1])
    ms['BG_margin'] = mae_margin
    ms['BG_alpha'] = alpha_k
    ms.to_csv('../../data 3/data_final_accu_cp_model3_BG.csv', index=False)