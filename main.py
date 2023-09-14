import os
from flask import Flask, request, render_template
import numpy as np
from PIL import Image

from sklearn.cluster import KMeans

app = Flask(__name__)
UPLOAD_FOLDER = 'static' # folder where the images are getting uploaded
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def extract_colors(image_path, num_colors=10):     # Adjust this to the desired number of colors
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img = img.resize((100, 100))
        pixel_values = np.array(img)
        # pixel_values = list(img.getdata())
    pixel_values = pixel_values.reshape(-1, 3)
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixel_values)

    # Get the RGB values of the cluster centers (dominant colors)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    # Count the number of pixels assigned to each cluster
    cluster_counts = np.bincount(kmeans.labels_)

    # Convert the RGB values to hex format
    colors = ['#%02x%02x%02x' % (r, g, b) for r, g, b in dominant_colors]
    # Calculate the percentage of each dominant color
    total_pixels = len(pixel_values)
    # percentages = (cluster_counts / total_pixels) * 100
    # Calculate the percentage of each dominant color with four decimal places
    percentages = [(count / total_pixels) for count in cluster_counts]
    percentages = [f"{percentage:.4f}" for percentage in percentages]
    return colors, percentages


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # image_path = None  # Initialize image_path
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('upload.html', error='No selected file')

        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            colors, percentages = extract_colors(filename)
            image_path = '/' + filename
            data = list(zip(colors, percentages))  # Combine colors and percentages
            return render_template('result.html', data=data, image_path=image_path)

    return render_template('upload.html')


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
