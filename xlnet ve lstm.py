from transformers import XLNetTokenizer, TFXLNetForSequenceClassification
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Veri kümesini yükle
df_train = pd.read_csv("train.csv", usecols=['text', 'target'])

# Veri kümesini train ve test olarak böl
train_data, test_data = train_test_split(df_train, test_size=0.2, random_state=42)

# XLNet tokenizatörünü yükle
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

# Girdileri ve etiketleri XLNet için uygun hale getir
train_encodings = tokenizer(train_data['text'].tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_data['text'].tolist(), truncation=True, padding=True)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_data['target'].values
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_data['target'].values
))

# XLNet modelini yükle
xlnet_model = TFXLNetForSequenceClassification.from_pretrained('xlnet-base-cased')

# XLNet modelinden çıktıları al
xlnet_inputs = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")
xlnet_outputs = xlnet_model(xlnet_inputs)[0]

# LSTM katmanı ekle
lstm_outputs = tf.keras.layers.LSTM(64)(xlnet_outputs)

# Sınıflandırma katmanı
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(lstm_outputs)

# Birleştirilmiş model
model = tf.keras.Model(inputs=xlnet_inputs, outputs=outputs)

# Modeli eğit
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

model.fit(train_dataset.shuffle(1000).batch(16), epochs=3, batch_size=16)

# Modelin performansını değerlendir
predictions = model.predict(test_dataset.batch(16))
predicted_labels = (predictions > 0.5).astype(int)

print(classification_report(test_data['target'], predicted_labels))
