from flask import Flask, render_template, request, jsonify
from inference import load_model_and_predict

app = Flask(__name__)

# Define the route for serving static files
app.static_folder = 'static'

@app.route('/')
def main_page():
    return render_template('index.html')

@app.route('/final')
def final_page():
    return render_template('final.html')


@app.route('/upload_image')
def image_upload_page():
    return render_template('final.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        image_path = 'uploaded_image.png'  # Specify the path to save the uploaded image
        file.save(image_path)
        predicted_class = load_model_and_predict(image_path)
        return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
