####### IMPORTS #############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from sklearn.metrics import confusion_matrix, classification_report
from plot_cm import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler

##### Loading saved csv ##############
df = pd.read_pickle("final_audio_data_csv/sri_audio_data_20221125.csv")

####### Making our data training-ready
X = df["feature"].values
X = np.concatenate(X, axis=0).reshape(len(X), 80)

y = np.array(df["class_label"].tolist())
y = to_categorical(y)

####### train test split ############
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

##### Training ############

# original 
model = Sequential([
    Dense(256, input_shape=X_train[0].shape),
    Activation('relu'),
    Dropout(0.5),
    Dense(256),
    Activation('relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# model = Sequential([
#     Dense(128, input_shape=X_train[0].shape),
#     Activation('relu'),
#     Dropout(0.5),
#     Dense(2, activation='softmax')
# ])

print(model.summary())

model.compile(
    loss="categorical_crossentropy",
    optimizer='adam',
    metrics=['accuracy']
)

print("Model Score: \n")
history = model.fit(X_train, y_train, epochs=1000)
model.save("saved_model/Sri_WWD_20221125.h5")
score = model.evaluate(X_test, y_test)
print(score)

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['accuracy'])
plt.legend(['loss','accuracy'])
plt.show()
#### Evaluating our model ###########
print("Model Classification Report: \n")
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(classification_report(np.argmax(y_test, axis=1), y_pred))
plot_confusion_matrix(cm, classes=["Does not have Wake Word", "Has Wake Word"])
