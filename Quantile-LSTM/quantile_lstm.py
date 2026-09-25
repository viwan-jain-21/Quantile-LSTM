import numpy
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no blocking plt.show()
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Layer, Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# pyrefly: ignore [missing-import]
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras import backend as K
from tensorflow import keras

period_size = 150
step_size = 50
median_q_threshold = 0.5
lower_q_threshold = 0.1
upper_q_threshold = 0.9

# ── Resolve dataset paths relative to this script's location ─────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_DATASET_DIR = os.path.join(_SCRIPT_DIR, '..', 'Datasets')

TRAIN_CSV    = os.path.join(_DATASET_DIR, 'speed_6005_train.csv')
LABELLED_CSV = os.path.join(_DATASET_DIR, 'speed_6005_labelled.csv')


# ── Custom Swish-like activation layer ───────────────────────────────────────
class Swish(Layer):
    """Soft-sign scaled by beta — used as the output activation."""

    def __init__(self, beta=1.5, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs):
        # softsign(x) * beta  (original paper intent)
        return K.softsign(inputs) * self.beta

    def get_config(self):
        config = {'beta': float(self.beta)}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


# ── Helper: list intersection ─────────────────────────────────────────────────
def intersection(lst1, lst2):
    lst3 = []
    for itr in range(len(lst1)):
        if lst1[itr][0] in lst2:
            lst3.append(lst1[itr])
    return lst3


# ── Stationarity check (ADF test) ─────────────────────────────────────────────
def verify_stationarity(dataset):
    is_stationary = True
    # adfuller expects a 1-D array
    test_results = adfuller(dataset.flatten(), result_object=False)

    print(f"ADF test statistic: {test_results[0]:.4f}")
    print(f"p-value:            {test_results[1]:.4f}")
    print("Critical thresholds:")
    for key, value in test_results[4].items():
        print(f"\t{key}: {value:.3f}")

    critical_1pct = list(test_results[4].values())[0]  # 1% level
    if test_results[0] > critical_1pct:
        print("Series is NON-STATIONARY")
        is_stationary = False
    else:
        print("Series is STATIONARY")
    return is_stationary


# ── Window → quantile features ────────────────────────────────────────────────
def create_dataset(dataset, look_back=1, tw=3):
    dataX, dataY = [], []
    dataUpperX, dataUpperY = [], []
    dataLowerX, dataLowerY = [], []
    multi = look_back // tw

    for i in range(len(dataset) - look_back - 1):
        q50X, q90X, q10X = [], [], []
        a = dataset[i + 1: i + look_back + 1]

        c = numpy.quantile(a, median_q_threshold)
        u = numpy.quantile(a, upper_q_threshold)
        l = numpy.quantile(a, lower_q_threshold)

        for j in range(0, len(a), tw):
            seg = a[j: j + tw]
            q50X.append(numpy.quantile(seg, median_q_threshold))
            q90X.append(numpy.quantile(seg, upper_q_threshold))
            q10X.append(numpy.quantile(seg, lower_q_threshold))

        dataX.append(q50X);   dataY.append(c)
        dataUpperX.append(q90X); dataUpperY.append(u)
        dataLowerX.append(q10X); dataLowerY.append(l)

    return (numpy.array(dataX),  numpy.array(dataY),
            numpy.array(dataUpperX), numpy.array(dataUpperY),
            numpy.array(dataLowerX), numpy.array(dataLowerY))


def identify_anomaly_quantiles(prediction_errors):
    anomaly_detection = []
    for m in range(0, len(prediction_errors), period_size):
        chunk = prediction_errors[m: m + period_size]
        upper_threshold = numpy.quantile(chunk, 0.9)
        lower_threshold = numpy.quantile(chunk, 0.1)
        for val in chunk:
            if (val > 0 and val > upper_threshold) or (val < 0 and val < lower_threshold):
                anomaly_detection.append(val)
    return anomaly_detection


def identify_anomaly(prediction_errors):
    anomaly_detection = []
    for m in range(0, len(prediction_errors), period_size):
        chunk = prediction_errors[m: m + period_size]
        avg  = numpy.average(chunk)
        std1 = numpy.std(chunk)
        upper_threshold = avg + 2 * std1
        lower_threshold = avg - 2 * std1
        for val in chunk:
            if val > upper_threshold or val < lower_threshold:
                anomaly_detection.append(val)
    return anomaly_detection


def identify_alpha(dataset):
    alpha_detection = []
    prev_slope = 1.0
    for m in range(0, len(dataset), period_size):
        chunk = dataset[m: m + period_size]
        slope = (chunk[-1] - chunk[0]) / period_size
        alpha = slope / prev_slope if prev_slope != 0 else 1.0
        alpha_detection.append(alpha)
        prev_slope = slope
    return float(numpy.absolute(numpy.mean(alpha_detection)))


# ── Build a Q-LSTM model ──────────────────────────────────────────────────────
def build_model(input_steps):
    """LSTM(4) → Dense(1) → Swish  (single output activation, no duplicate layer)."""
    model = Sequential([
        LSTM(4, input_shape=(input_steps, 1)),
        Dense(1),
        Swish(beta=1.5),          # single Swish layer -- removes the old bug of
                                  # stacking Swish() + Activation(Swish()) together
    ])
    # 'logcosh' string removed in Keras 3 — use the class directly
    model.compile(loss=keras.losses.LogCosh(), optimizer='adam')
    return model


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # ── Reproducibility ──────────────────────────────────────────────────────
    numpy.random.seed(7)

    # ── Load training data (speed_6005_train.csv) ────────────────────────────
    # Columns: <index>, timestamp, value, label  →  usecols=[2] = 'value'
    dataframe = read_csv(TRAIN_CSV, usecols=[2], engine='python')
    dataset   = dataframe.values.astype('float32')

    print("\n=== Stationarity check (training set) ===")
    verify_stationarity(dataset)

    # ── Normalise ─────────────────────────────────────────────────────────────
    scaler  = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # ── Train / validation split (70 / 30) ───────────────────────────────────
    train_size = int(len(dataset) * 0.7)
    train, test = dataset[:train_size], dataset[train_size:]
    print(f"Total samples: {len(dataset)}  |  Train: {len(train)}  |  Val: {len(test)}")

    # ── Build windowed feature arrays ─────────────────────────────────────────
    look_back = period_size       # 150 time-steps per window
    tw        = step_size         # 50-step sub-windows
    multi     = look_back // tw   # number of LSTM time-steps

    (trainX,  trainY,
     trainXU, trainYU,
     trainXL, trainYL) = create_dataset(train, look_back, tw)

    (testX,  testY,
     testXU, testYU,
     testXL, testYL) = create_dataset(test, look_back, tw)

    # ── Reshape to (samples, time_steps, features=1) ──────────────────────────
    def r3d(arr):
        return arr.reshape(arr.shape[0], arr.shape[1], 1)

    trainX  = r3d(trainX);  testX  = r3d(testX)
    trainXU = r3d(trainXU); testXU = r3d(testXU)
    trainXL = r3d(trainXL); testXL = r3d(testXL)

    # ── Visualise training series ─────────────────────────────────────────────
    plt.figure(figsize=(12, 4))
    plt.plot(train, label='Training data (speed_6005)')
    plt.title('speed_6005 — training segment (normalised)')
    plt.xlabel('Time step'); plt.ylabel('Normalised speed')
    plt.legend(loc='best'); plt.tight_layout()
    plt.savefig(os.path.join(_SCRIPT_DIR, 'train_series.png'), dpi=100)
    # plt.show() omitted — Agg backend saves to file only

    alpha = identify_alpha(dataset.flatten())
    print(f'\nAlpha (trend ratio): {alpha:.4f}')

    # ── Train three Q-LSTM models ─────────────────────────────────────────────
    print("\n=== Training Q50 (median) model ===")
    modelq50 = build_model(multi)
    modelq50.fit(trainX,  trainY,  epochs=50, batch_size=1, verbose=2)

    print("\n=== Training Q10 (lower) model ===")
    modelq10 = build_model(multi)
    modelq10.fit(trainXL, trainYL, epochs=50, batch_size=1, verbose=2)

    print("\n=== Training Q90 (upper) model ===")
    modelq90 = build_model(multi)
    modelq90.fit(trainXU, trainYU, epochs=50, batch_size=1, verbose=2)

    # ── Load full labelled dataset for inference ──────────────────────────────
    labelled_df   = read_csv(LABELLED_CSV, engine='python')
    # Ground-truth anomaly row indices (label == 1)
    anomaly_indices = set(labelled_df.index[labelled_df['label'] == 1].tolist())
    total_points    = len(labelled_df)
    print(f"\nTrue anomaly row indices: {anomaly_indices}  (count={len(anomaly_indices)})")

    # Scale using the same scaler fitted on training
    full_scaled = scaler.transform(labelled_df[['value']].values.astype('float32'))

    # ── Sliding-window inference ──────────────────────────────────────────────
    i, j = 0, look_back
    detected_indices = []   # row indices flagged as anomalies

    while j < len(full_scaled):
        temp = full_scaled[i:j]

        q50_arr, q10_arr, q90_arr = [], [], []
        for m in range(0, len(temp), tw):
            seg = temp[m: m + tw]
            q50_arr.append([float(numpy.quantile(seg, median_q_threshold))])
            q10_arr.append([float(numpy.quantile(seg, lower_q_threshold))])
            q90_arr.append([float(numpy.quantile(seg, upper_q_threshold))])

        # Wrap in a batch dimension → shape (1, multi, 1)
        X50 = numpy.array([q50_arr])
        X10 = numpy.array([q10_arr])
        X90 = numpy.array([q90_arr])

        q50_pred = modelq50.predict(X50, verbose=0)
        q10_pred = modelq10.predict(X10, verbose=0)
        q90_pred = modelq90.predict(X90, verbose=0)

        next_idx = j
        if next_idx < len(full_scaled):
            next_val = float(full_scaled[next_idx, 0])
            iqr = float(q90_pred[0, 0]) - float(q10_pred[0, 0])
            ucl = float(q50_pred[0, 0]) + 0.9 * iqr
            lcl = float(q50_pred[0, 0]) - 0.9 * iqr
            if next_val > ucl or next_val < lcl:
                detected_indices.append(next_idx)

        i += 1
        j += 1

    # ── Evaluation metrics (index-based) ─────────────────────────────────────
    detected_set = set(detected_indices)
    tp = len(detected_set & anomaly_indices)
    fp = len(detected_set - anomaly_indices)
    fn = len(anomaly_indices - detected_set)
    tn = total_points - tp - fp - fn

    print("\n=== Anomaly Detection Results (speed_6005) ===")
    print(f"Total data points : {total_points}")
    print(f"True anomalies    : {len(anomaly_indices)}  -> indices {anomaly_indices}")
    print(f"Detected anomalies: {len(detected_set)}")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    # ── Final plot: full series + detected anomalies ──────────────────────────
    plt.figure(figsize=(14, 5))
    plt.plot(full_scaled, label='speed_6005 (normalised)', linewidth=0.8, color='steelblue')

    if detected_indices:
        plt.scatter(detected_indices,
                    full_scaled[detected_indices],
                    color='red', zorder=5, s=30, label='Detected anomaly')

    for idx in anomaly_indices:
        plt.axvline(x=idx, color='orange', linestyle='--', linewidth=1.4,
                    label='True anomaly' if idx == list(anomaly_indices)[0] else '')

    plt.title('Quantile-LSTM Anomaly Detection — speed_6005')
    plt.xlabel('Time step'); plt.ylabel('Normalised speed')
    plt.legend(loc='upper right'); plt.tight_layout()
    plt.savefig(os.path.join(_SCRIPT_DIR, 'anomaly_detection_speed6005.png'), dpi=100)
    # plt.show() omitted — Agg backend saves to file only