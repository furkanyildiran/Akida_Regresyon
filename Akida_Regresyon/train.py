import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import models, layers
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from quantizeml.models import quantize, QuantizationParams
from cnn2snn import convert, set_akida_version, AkidaVersion
from akida import devices

# Veri setini yükle
df = pd.read_excel('capacityFade.xlsx')
df = df[['Cap_2000', 'Cap_3500']].dropna()

# Ölçeklendirme öncesi ham veriyi tut
data_2000 = df['Cap_2000'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
data_2000_scaled = scaler.fit_transform(data_2000)

# Veri setini oluştur
def create_dataset(dataset, look_back=5):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X).reshape(-1, look_back, 1), np.array(Y)

look_back = 5
train_size = 1950
X_scaled, Y = create_dataset(data_2000_scaled, look_back)
X_uint8 = (data_2000_scaled * 255).astype(np.uint8)
X, _ = create_dataset(X_uint8, look_back)

X_train_scaled, Y_train = X_scaled[:train_size], Y[:train_size]
X_test_scaled, Y_test = X_scaled[train_size:], Y[train_size:]
X_train_uint8 = X[:train_size]
X_test_uint8 = X[train_size:]

# 4D input for Conv2D
X_train_scaled = X_train_scaled.reshape(-1, look_back, 1, 1)
X_test_scaled = X_test_scaled.reshape(-1, look_back, 1, 1)
X_train_uint8 = X_train_uint8.reshape(-1, look_back, 1, 1)
X_test_uint8 = X_test_uint8.reshape(-1, look_back, 1, 1)

model_keras = models.Sequential([
    layers.Rescaling(1. / 255, input_shape=(look_back, 1, 1)),
    layers.Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='valid'),
    layers.ReLU(max_value=15),
    layers.Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='valid'),
    layers.ReLU(max_value=15),
    layers.Flatten(),
    layers.Dense(1)
], name='capacity_prediction')

model_keras.summary()


model_keras.compile(
    loss=MeanSquaredError(),
    optimizer=Adam(learning_rate=1e-3),
    metrics=['mae']
)
history = model_keras.fit(X_train_scaled, Y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)


score = model_keras.evaluate(X_test_scaled, Y_test, verbose=0)
print('Test MAE (original):', score[1])


#Because Akida 1.0 has 4 bit weights, 4 bit quantization must be applied.
qparams = QuantizationParams(
    input_weight_bits=8,
    weight_bits=4,
    activation_bits=4
)
model_quantized = quantize(model_keras, qparams=qparams)

def compile_evaluate(model):
    model.compile(loss=MeanSquaredError(), metrics=['mae'])
    return model.evaluate(X_test_scaled, Y_test, verbose=0)[1]

print('Test MAE after 4-bit quantization:', compile_evaluate(model_quantized))

# Calibration
model_quantized = quantize(model_keras, qparams=qparams,
                           samples=X_train_scaled, num_samples=1024, batch_size=100, epochs=2)
print('Test MAE after 4-bit calibration:', compile_evaluate(model_quantized))


model_quantized.compile(
    loss=MeanSquaredError(),
    optimizer=Adam(learning_rate=1e-5),
    metrics=['mae']
)
model_quantized.fit(X_train_scaled, Y_train, epochs=30, validation_split=0.1, verbose=1)
score = model_quantized.evaluate(X_test_scaled, Y_test, verbose=0)[1]
print('Test MAE after fine tuning:', score)


with set_akida_version(AkidaVersion.v1):
    model_akida = convert(model_quantized)

model_akida.summary()
model_akida.save("./model_akida.akida")

# Akida ile tahmin yap ve MAE hesapla
predictions = model_akida.predict(X_test_uint8)
predictions = predictions / 255.0  # 0-1 ölçeğine çevir

# Tahminleri ve gerçek değerleri orijinal ölçeğe çevir
predictions_original = scaler.inverse_transform(predictions.reshape(-1, 1))
Y_test_original = scaler.inverse_transform(Y_test.reshape(-1, 1))

# Orijinal ölçekte MAE
mae_original = np.mean(np.abs(predictions.squeeze() - Y_test))#predictions_original - Y_test_original))
print('Test MAE after conversion :', mae_original)

# İlk 5 tahmini ve gerçek değeri karşılaştır (orijinal ölçekte)
print("First 5 predictions (original scale):", predictions_original[:5].flatten())
print("First 5 true values (original scale):", Y_test_original[:5].flatten())
