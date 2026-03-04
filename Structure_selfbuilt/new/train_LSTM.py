import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# === 1. 读取并准备数据 ===
df = pd.read_csv(r"D:\Anaconda3\Quant\Stock_Quant\Structure_selfbuilt\data\market_states_601788_SH_with_labels.csv", encoding="utf-8-sig")

# 特征和标签
X = df[['hmm_p0', 'hmm_p1', 'hmm_p2', 'hmm_p3', 'hmm_p4']].values
y = df['y'].values

# 数据切分
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

# === 2. 计算类别权重 ===
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# === 3. 构建 LSTM 模型 ===
def build_lstm_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # LSTM layer
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    
    # Dense layer and output layer
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === 4. 模型训练 ===
model = build_lstm_model((X_train.shape[1], 1))

# Reshape the input for LSTM (for single feature in each timestep)
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train the model
history = model.fit(X_train_reshaped, y_train, validation_data=(X_val_reshaped, y_val), 
                    epochs=50, batch_size=64, class_weight=class_weight_dict)

# === 5. 模型评估 ===
loss, accuracy = model.evaluate(X_test_reshaped, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# === 6. 预测 ===
y_pred = model.predict(X_test_reshaped)
y_pred = (y_pred > 0.5).astype(int)

# Print classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
