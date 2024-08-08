from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import numpy as np
import pandas as pd

app = Flask(__name__)

# Muat model
model = tf.keras.models.load_model('model.h5')

@app.route('/', methods=['GET'])
def index():
    # Tampilkan halaman HTML untuk menginput data
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Dapatkan data dari request POST
    array_string = request.form.get('input_name')
    array_string = array_string.replace('[', '').replace(']', '').split()

    if array_string is None:
        return render_template('index.html', prediction_text='No data provided')
    
    means = pd.read_csv('path_to_means.csv').iloc[:, 1]
    std_devs = pd.read_csv('path_to_std_devs.csv').iloc[:, 1]
    # means = pd.read_csv('path_to_means.csv', header=None, usecols=[1], squeeze=True)
    # std_devs = pd.read_csv('path_to_std_devs.csv', header=None, usecols=[1], squeeze=True)


    # Buat menjadi numpy array dan hapus kolom 3, 64, 65
    input_data = np.array([float(x) for x in array_string])
    input_data = np.delete(input_data, [3, 64, 65])

    # Normalisasi data baru
    input_data = (input_data - means.values) / std_devs.values

    # Sesuaikan shape untuk model
    input_data = input_data.reshape(-1, 125, 1)
    
    # Proses dan prediksi
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)

    return render_template('index.html', prediction_text=f'Kelas Prediksi: {predicted_class[0]}')

if __name__ == '__main__':
    app.run(debug=True)
