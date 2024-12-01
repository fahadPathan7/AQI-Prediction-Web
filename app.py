from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image as im
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import sys
from lime import lime_image

# Suppress oneDNN custom operations messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress TensorFlow CPU feature guard messages
tf.get_logger().setLevel('ERROR')

# Constants
MODEL_PATH = os.path.join(os.getcwd(), "model.h5")

# Preprocess image function
def preprocess_image(image):
    # Resize the image to 200x200
    image = tf.image.resize(image, (200, 200))

    # Ensure the image has 3 channels
    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    elif image.shape[-1] != 3:
        image = tf.expand_dims(image, axis=-1)
        image = tf.image.grayscale_to_rgb(image)

    # Normalize the image
    image = image / 255.0

    # Crop the image to the first 120 rows
    cropped_image = image[:120]

    # Ensure the image has the correct shape (120, 200, 3)
    cropped_image = tf.ensure_shape(cropped_image, (120, 200, 3))

    return cropped_image

# Load the model
loaded_model = tf.keras.models.load_model(MODEL_PATH)

# Recompile the model to address optimizer warning
loaded_model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_squared_error', tf.keras.metrics.RootMeanSquaredError()])

# Define the prediction function for LIME
def predict_fn(images):
    preds = loaded_model.predict(images)
    return preds

# Initialize LIME explainer
explainer = lime_image.LimeImageExplainer()

# Create a Flask app
app = Flask(__name__)

# Home route
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Load the uploaded image
    imagefile = request.files['imagefile']
    image_filename = imagefile.filename
    image_path = os.path.join("static/images/", image_filename)

    # Ensure the directory exists and has write permissions
    if not os.path.exists("static/images/"):
        os.makedirs("static/images/")
    os.chmod("static/images/", 0o777)

    # Delete all files in the static/images folder
    files = os.listdir('static/images/')
    for file in files:
        os.remove(os.path.join('static/images/', file))

    imagefile.save(image_path)

    # Load and preprocess the image
    uploaded_image = im.open(image_path)
    uploaded_image = np.array(uploaded_image)
    preprocessed_image = preprocess_image(uploaded_image)

    # Expand dimensions to match the expected input shape (batch size of 1)
    preprocessed_image_expanded = tf.expand_dims(preprocessed_image, axis=0)

    # Make predictions using the loaded model
    prediction = loaded_model.predict(preprocessed_image_expanded)

    # Extract the prediction value and convert it to an integer
    prediction_value = int(prediction[0][0])

    # Convert the preprocessed image to a NumPy array for LIME
    preprocessed_image_np = preprocessed_image.numpy()

    # Generate LIME explanation
    explanation = explainer.explain_instance(
        preprocessed_image_np,  # The preprocessed image to explain
        predict_fn,   # Prediction function
        top_labels=1,  # Number of top labels to explain
        hide_color=0,  # The color for the masked parts
        num_samples=1000  # Number of samples to generate
    )

    # Get the explanation for the top label
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=10,
        hide_rest=False
    )

    plt.imshow(mark_boundaries(temp, mask))
    # plt.title(f'LIME Explanation')
    lime_plot_path = os.path.join("static/images/", "lime_plot.png")
    plt.savefig(lime_plot_path)
    plt.close()

    # Return the prediction result
    return render_template('index.html', imagefile=image_filename, prediction=prediction_value, lime_plot_path=lime_plot_path)

# Run the app
if __name__ == '__main__':
    app.run(port=5000)