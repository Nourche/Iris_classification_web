from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess data (run only once)
file_path = 'iris.data'  # Path to your dataset
data = pd.read_csv(file_path, header=None, sep=',')
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Encode the target labels
label_encoder = LabelEncoder()
data['species_encoded'] = label_encoder.fit_transform(data['species'])

# Separate features and target
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = data['species_encoded'].values

# Train the LDA model
lda_classifier = LDA()
lda_classifier.fit(X, y)

# Route: Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Route: Predict
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        # Predict the class
        example = [[sepal_length, sepal_width, petal_length, petal_width]]
        predicted_class = lda_classifier.predict(example)
        class_name = label_encoder.inverse_transform(predicted_class)[0]

        return jsonify({'prediction': class_name})
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
