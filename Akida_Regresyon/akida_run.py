import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import models, layers
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from quantizeml.models import quantize, QuantizationParams
from cnn2snn import convert, set_akida_version, AkidaVersion
from akida import devices
from akida import Model

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

Y_test_original = scaler.inverse_transform(Y_test.reshape(-1,1))

model_akida = Model("./model_akida.akida")
model_akida.summary()


# Hardware mapping
device = devices()[0]
print(f"Using device: {device}")
model_akida.map(device)
device.soc.power_measurement_enabled = True
model_akida.summary()

# Test a single example (orijinal ölçekte)
sample_image = 0
image = X_test_uint8[sample_image]
output = model_akida.forward(image.reshape(1, look_back, 1, 1))
output = output / 255.0  # 0-1 ölçeğine çevir
output_original = scaler.inverse_transform(output.reshape(-1, 1))
print(f"Input sequence (original scale): {scaler.inverse_transform(X_test_scaled[sample_image].squeeze().reshape(-1, 1)).flatten()}")
print(f"True value (original scale): {Y_test_original[sample_image, 0]:.4f}")
print(f"Predicted value (original scale): {output_original[0, 0]:.4f}")
floor_power = device.soc.power_meter.floor
print(f'Floor power : {floor_power:.2f} mW')