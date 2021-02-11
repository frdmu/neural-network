from tensorflow import keras
from tensorflow.keras import layers

input_shape=
output_shape=

model = keras.Sequential([
	layers.BatchNormalization(input_shape=input_shape),	

	layers.Dense(units=1024, activation='relu'),
	layers.Dropout(0.3),
	layers.BatchNormalization(),	

	layers.Dense(units=1024, activation='relu'),
	layers.Dropout(0.3),
	layers.BatchNormalization(),	

	layers.Dense(units=1024, activation='relu'),
	layers.Drop(0.3),
	layers.BatchNormalization(),	

	layers.Dense(units=output_shape),
	])

model.compile(optimizer='adam', loss='mae', metrics=[mae])

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
	X_train, y_train,
	validation_data=(X_valid, y_valid),
	batch_size= ,
	epochs=100,
	callbacks=[early_stopping],
	verbose=0,# hide the output because we have so many epochs
)


history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
