import pandas as pd
import numpy as np
import skimage


def patient_extr(df):
    sirt = pd.concat([df.loc[:, 'SIRT_0':'SIRT_14'], df.loc[:, 'SIRT_20':'SIRT_34']], axis=1).values
    chrt = df.loc[:, 'CHRT_0':'CHRT_29'].values
    nbk1 = df.loc[:, 'NBK1_1':'NBK1_30'].values
    assert sirt.shape[1] == 30
    assert chrt.shape[1] == 30
    assert nbk1.shape[1] == 30

    return sirt, chrt, nbk1


def find_maxmin(data, persentage='2%'):
    sorted_data = np.sort(data.flatten())
    if persentage == '2%':
        cutoff_min = sorted_data[round(len(sorted_data) * 0.02): round(len(sorted_data) * 0.98)][0]
        cutoff_max = sorted_data[round(len(sorted_data) * 0.02): round(len(sorted_data) * 0.98)][-1]
    else:
        raise Exception
    return cutoff_min, cutoff_max


def minmax_norm(data, pixel_min, pixel_max):
    ymin, ymax = 0, 255
    for j in range(data.shape[0]):
        for k in range(data.shape[1]):
            data[j][k] = round(((ymax - ymin) * (data[j][k] - pixel_min) / (pixel_max - pixel_min)) + ymin)
    return data


def task_norm(slideimage):
    max_value = np.max(slideimage)
    slideimage[slideimage < 0] = max_value
    cutoff_min, cutoff_max = find_maxmin(slideimage, persentage='2%')
    slideimage[slideimage < cutoff_min] = cutoff_min
    slideimage[slideimage > cutoff_max] = cutoff_max
    slideimage_normed = minmax_norm(slideimage, cutoff_min, cutoff_max)
    return slideimage_normed


def image_normalization(wholeimage):
    normed_data = []
    for i in range(wholeimage.shape[0]):
        normed_data.append(task_norm(wholeimage[i, :, :]))
    normed_img = np.stack((normed_data[0], normed_data[1], normed_data[2]), axis=0)
    assert normed_img.shape == wholeimage.shape
    return normed_img


if __name__ == '__main__':
    T_step = 6  # 6 months
    Tmax = 96  # largest time
    mae_margin_how = 'toTmax'
    dir_imgsave = '../data/img/'

    # read msbase dataset
    msb_dataset = pd.read_csv('../../data 3/data_final_accu_cp_model3.csv')
    msb_dataset['last_visit'] = pd.to_datetime(msb_dataset['last_visit'])
    msb_dataset['last_visit'] = msb_dataset['last_visit'].dt.to_period('M')
    base_patient = msb_dataset['PATIENT_ID'].unique()

    # mae_margin, _, _ = mae_margin_info(msb_dataset['stop'], msb_dataset['event_accu'], how=mae_margin_how)  # don't need extend image to mae_margin timepoint

    # read msreactor dataset
    msr_dataset = pd.read_csv('../data/MSREACTOR_model3.csv')
    msr_dataset = msr_dataset.applymap(
        lambda x: x.lstrip('fposneg:') if type(x) == str else x)  # delete strings in tabular
    msr_dataset = msr_dataset.applymap(lambda x: int(0) if x == '-' else x)  # change all "-" to 0
    msr_dataset = msr_dataset.apply(pd.to_numeric, errors='ignore')
    # select last test in each month for individuals
    msr_dataset['testdate'] = pd.to_datetime(msr_dataset['testdate'])
    msr_dataset['testdate'] = msr_dataset['testdate'].dt.to_period('M')
    msr_dataset = msr_dataset.groupby('PATIENT_ID', as_index=False).apply(
        lambda df: df.drop_duplicates('testdate', keep='last'))

    for idx, patient_id in enumerate(base_patient):
        patient = msr_dataset[msr_dataset['PATIENT_ID'] == patient_id]
        lastvisit_date = msb_dataset[msb_dataset['PATIENT_ID'] == patient_id]['last_visit']
        testdate = pd.concat((patient['testdate'], lastvisit_date))
        testrepeat = testdate.astype(int).diff().iloc[1:].values
        testrepeat[-1] += 1  # survival/censoring month should have time-varying features

        s, c, n = patient_extr(patient)
        piled_data = np.stack((s, c, n), axis=0)
        impulted_data = np.repeat(piled_data, testrepeat.astype(int),
                                  axis=1)  # in image manner already, w*h*c, imputed_data[:,:,0] will be repeated sirt
        img = image_normalization(impulted_data)
        img_padded = np.concatenate((img, np.zeros((img.shape[0], Tmax - img.shape[1], img.shape[2]))),
                                    axis=1)  # padding zero after survival/censoring time
        img_swapped = np.transpose(img_padded, (1, 2, 0))
        skimage.io.imsave(dir_imgsave + patient_id + '.png', img_swapped.astype(np.uint8))
