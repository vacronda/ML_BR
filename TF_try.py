import pandas as pd 
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv(r'data.csv')
#dataset = dataset.astype('float32')
is_train = dataset['train']==1
is_test = dataset['test']==1
is_previsao = dataset['previsao']==1
data_train = dataset[is_train]
data_test = dataset[is_test]
data_previsao = dataset[is_previsao]
#x_train = data_train.filter(items=['preco_home_10' , 'preco_away_10', 'home_pontos_a5', 'away_pontos_a5', 'home_saldo_a5', 'away_saldo_a5'])
#x_test = data_test.filter(items=['preco_home_10' , 'preco_away_10', 'home_pontos_a5', 'away_pontos_a5', 'home_saldo_a5', 'away_saldo_a5'])
#x_train = data_train.filter(items=['preco_home_10' , 'preco_away_10'])
# #x_test = data_test.filter(items=['preco_home_10' , 'preco_away_10'])

x_train = data_train.filter(items=['preco_home_10' , 'preco_away_10','home_golspro_a5', ' away_golspro_a5', ' home_golscon_a5', ' away_golscon_a5', ' home_vitoria_a5', ' away_vitoria_a5', ' home_pontos_a5', ' away_pontos_a5', ' home_saldo_a5', ' away_saldo_a5', ' c_pontosmedio_t10_home_res', ' c_preco_t10_home_res', ' c_pontosmedio_t10_away_res', ' c_preco_t10_away_res'])
x_test = data_test.filter(items=['preco_home_10' , 'preco_away_10','home_golspro_a5', ' away_golspro_a5', ' home_golscon_a5', ' away_golscon_a5', ' home_vitoria_a5', ' away_vitoria_a5', ' home_pontos_a5', ' away_pontos_a5', ' home_saldo_a5', ' away_saldo_a5', ' c_pontosmedio_t10_home_res', ' c_preco_t10_home_res', ' c_pontosmedio_t10_away_res', ' c_preco_t10_away_res'])
x_previsao = data_previsao.filter(items=['preco_home_10' , 'preco_away_10','home_golspro_a5', ' away_golspro_a5', ' home_golscon_a5', ' away_golscon_a5', ' home_vitoria_a5', ' away_vitoria_a5', ' home_pontos_a5', ' away_pontos_a5', ' home_saldo_a5', ' away_saldo_a5', ' c_pontosmedio_t10_home_res', ' c_preco_t10_home_res', ' c_pontosmedio_t10_away_res', ' c_preco_t10_away_res'])

y_train = data_train["vencedor"]
y_test= data_test["vencedor"]



x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
x_previsao = tf.keras.utils.normalize(x_previsao, axis=1)

#model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Flatten())
#model.add(tf.keras.layers.Dense(50, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(50, activation=tf.nn.softmax))
#model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
#model.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])
# model.fit(x_train.values, y_train.values, epochs=100)


## Acc=58 pontos =609
#model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Flatten())
#model.add(tf.keras.layers.Dense(24, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(24, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
#model.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])


#model_history = model.fit(x_train.values, y_train.values,
#                                  epochs=100,
#                                  batch_size=512,
#                                  validation_data=(x_test.values, y_test.values),
#                                  verbose=2)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(516, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(516, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model_history = model.fit(x_train.values, y_train.values,
                                  epochs=100,
                                  batch_size=512,
                                  validation_data=(x_test.values, y_test.values),
                                  verbose=2)


val_loss, val_acc = model.evaluate(x_test.values, y_test.values)
print(val_loss)
print(val_acc)

model.save('epic_num_reader.model')

new_model = tf.keras.models.load_model('epic_num_reader.model')

predictions_test = new_model.predict(x_test.values)

df=pd.DataFrame(predictions_test,data_test["vencedor"])

df.to_excel(r'results\predictions_test.xlsx')

predictions_prev = new_model.predict(x_previsao.values)

df2=pd.DataFrame(predictions_prev,data_previsao[["home_team","away_team"]])

df2.to_excel(r'results\predictions_prev.xlsx')

print(df2)

