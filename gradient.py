import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from nolds import hurst_rs
from sklearn.metrics import make_scorer, roc_auc_score, f1_score, roc_curve, confusion_matrix
from scipy.stats import skew, kurtosis
from entropy_and_complexity import get_entropy_and_complexity_sorting
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from fft import fft, fft_2, max_amplitude


def find_parameters(series, rates):
    data = []

    for i in range(len(series)):
        H, C = get_entropy_and_complexity_sorting(series[i])
        Hu = hurst_rs(series[i])
        Sk = skew(series[i])
        Ku = kurtosis(series[i])
        A = max_amplitude(series[i], rates[i])
        data.append((H, C, Hu, Sk, Ku, A))

    return np.array(data)


def gradient_boosting(type=roc_auc_score):
    conclusions = np.load("conclusions.npy")

    data_regular = []
    data_chaotic = []
    rates_regular = []
    rates_chaotic = []

    for i in range(39):
        wav = wavfile.read(f'wavs/rec{i}.wav')
        rate = wav[0]
        current_file = np.array(wav[1])
        current_file = fft(current_file, 100, 600, rate)
        k = current_file.shape[0]

        if conclusions[i] == '':
            for j in range(9):
                rates_regular.append(rate)
            data_regular.append(current_file[(3 * k // 27):(4 * k // 27)])
            data_regular.append(current_file[(4 * k // 27):(5 * k // 27)])
            data_regular.append(current_file[(5 * k // 27):(6 * k // 27)])
            data_regular.append(current_file[(12 * k // 27):(13 * k // 27)])
            data_regular.append(current_file[(13 * k // 27):(14 * k // 27)])
            data_regular.append(current_file[(14 * k // 27):(15 * k // 27)])
            data_regular.append(current_file[(21 * k // 27):(22 * k // 27)])
            data_regular.append(current_file[(22 * k // 27):(23 * k // 27)])
            data_regular.append(current_file[(23 * k // 27):(24 * k // 27)])
        else:
            for j in range(9):
                rates_chaotic.append(rate)
            data_chaotic.append(current_file[(3 * k // 27):(4 * k // 27)])
            data_chaotic.append(current_file[(4 * k // 27):(5 * k // 27)])
            data_chaotic.append(current_file[(5 * k // 27):(6 * k // 27)])
            data_chaotic.append(current_file[(12 * k // 27):(13 * k // 27)])
            data_chaotic.append(current_file[(13 * k // 27):(14 * k // 27)])
            data_chaotic.append(current_file[(14 * k // 27):(15 * k // 27)])
            data_chaotic.append(current_file[(21 * k // 27):(22 * k // 27)])
            data_chaotic.append(current_file[(22 * k // 27):(23 * k // 27)])
            data_chaotic.append(current_file[(23 * k // 27):(24 * k // 27)])

    params_regular = find_parameters(data_regular, rates_regular)
    params_chaotic = find_parameters(data_chaotic, rates_chaotic)

    result_regular = np.zeros(len(data_regular))
    result_chaotic = np.ones(len(data_chaotic))

    x_train_regular, x_test_regular, y_train_regular, y_test_regular = train_test_split(params_regular, result_regular,
                                                                                        test_size=0.25)
    x_train_chaotic, x_test_chaotic, y_train_chaotic, y_test_chaotic = train_test_split(params_chaotic, result_chaotic,
                                                                                        test_size=0.25)

    x_train = np.concatenate([x_train_regular, x_train_chaotic])
    x_test = np.concatenate([x_test_regular, x_test_chaotic])
    y_train = np.concatenate([y_train_regular, y_train_chaotic])
    y_test = np.concatenate([y_test_regular, y_test_chaotic])

    params = {'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1], 'n_estimators': [25, 50, 100, 150, 200, 300],
              'max_depth': [3, 5, 7], 'min_samples_leaf': [1, 2, 3, 4, 5, 7, 10]}
    model = GridSearchCV(GradientBoostingClassifier(), cv=3, param_grid=params,
                         scoring=make_scorer(type))
    model.fit(x_train, y_train)

    y_score = model.decision_function(x_test)
    y_pred = model.predict(x_test)

    score = model.score(x_test, y_test)
    roc_score = roc_auc_score(y_test, y_score)
    f_score = f1_score(y_test, y_pred, average='weighted')

    print(f'Test score: {score}')
    print(f'ROC AUC score: {roc_score}')
    print(f'F1 score: {f_score}')
    print(y_pred)

    false_positive, true_positive, _ = roc_curve(y_test, y_score)
    plt.plot(false_positive, true_positive, label=f'AUC = {roc_score}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    matrix = confusion_matrix(y_test, y_pred)

    sns.heatmap(matrix, annot=True, fmt='d', cmap='Reds', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Data')
    plt.show()

    return roc_score, f_score


def gradient_test(type=roc_auc_score):
    roc_average = 0
    f_average = 0

    for i in range(5):
        current_roc, current_f = gradient_boosting(type)
        roc_average += current_roc
        f_average += current_f

    print(roc_average / 5, f_average / 5)
