import os
import numpy as np
import pickle


def normalize(data):
    '''
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    data = min_max_scaler.fit_transform(data)
    '''
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)

    return data

def seed_data():
    EEG_dir = '/data2/ch/emotion/data/SEED_Multimodal/Chinese/02-EEG-DE-feature/eeg_used_4s'
    eye_dir = '/data2/ch/emotion/data/SEED_Multimodal/Chinese/04-Eye-tracking-feature/eye_tracking_feature'
    eye_all_sub = list()
    eeg_all_sub = list()
    label_all_sub = list()
    for sub in [1,2,3,4,5,8,9,10,11,12,13,14]:
        eeg_sub = []
        eye_sub = []
        label_sub = []
        for section in range(1, 4):
            eye_file = os.path.join(eye_dir, str(sub) + '_' + str(section))
            eye_data = pickle.load(open(eye_file, 'rb'))
            eye_data = np.vstack([eye_data['train_data_eye'], eye_data['test_data_eye']])
            eye_data = normalize(eye_data)


            eeg_data = np.load(os.path.join(EEG_dir, str(sub) + '_' + str(section) + '.npz'))
            train_data_eeg = pickle.loads(eeg_data['train_data'])
            test_data_eeg = pickle.loads(eeg_data['test_data'])

            label = np.hstack([eeg_data['train_label'], eeg_data['test_label']])

            train_data_all_bands = []
            test_data_all_bands = []

            for bands in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                train_tmp = train_data_eeg[bands]
                test_tmp = test_data_eeg[bands]
                if bands == 'delta':
                    train_data_all_bands = train_tmp
                    test_data_all_bands = test_tmp
                else:
                    train_data_all_bands = np.hstack((train_data_all_bands, train_tmp))
                    test_data_all_bands = np.hstack((test_data_all_bands, test_tmp))

            eeg_data = np.vstack([train_data_all_bands, test_data_all_bands])
            eeg_data = normalize(eeg_data)


            if section == 1:
                eeg_sub = eeg_data
                eye_sub = eye_data
                label_sub = label
            else:
                eeg_sub = np.vstack([eeg_sub, eeg_data])
                eye_sub = np.vstack([eye_sub, eye_data])
                label_sub = np.hstack([label_sub, label])

        eye_all_sub.append(eye_sub)
        eeg_all_sub.append(eeg_sub)
        label_all_sub.append(label_sub)

    eye_all_sub = np.array(eye_all_sub)
    eeg_all_sub = np.array(eeg_all_sub)
    label_all_sub = np.array(label_all_sub)

    return eeg_all_sub, eye_all_sub, label_all_sub


