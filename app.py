import os
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
from rag import query_retro_rag
from fetalclass import run_segmentation  # Import the run_segmentation function

app = Flask(__name__)

# Folder where uploaded images will be stored
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure the folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Path for storing form data
FORM_DATA_FILE = 'form_data.txt'

# Route for rendering the chatbot form
@app.route('/')
def index():
    return render_template('chatbot.html')

# Route for saving form data to a text file and proceeding to chatbot
@app.route('/save-form', methods=['POST'])
def save_form():
    # Collect the form data
    status = request.form.get('status')
    trimester = request.form.get('trimester', '')
    symptoms = request.form.get('symptoms', '')
    support = request.form.get('support', '')
    delivery_time = request.form.get('deliveryTime', '')
    postpartum_symptoms = request.form.get('postpartumSymptoms', '')
    postpartum_support = request.form.get('postpartumSupport', '')

    # Write the form data to a text file
    with open(FORM_DATA_FILE, 'a') as file:
        file.write(f"Status: {status}\n")
        if status == 'pregnant':
            file.write(f"Trimester: {trimester}\n")
            file.write(f"Symptoms: {symptoms}\n")
            file.write(f"Support: {support}\n")
        elif status == 'delivered':
            file.write(f"Delivery Time: {delivery_time}\n")
            file.write(f"Postpartum Symptoms: {postpartum_symptoms}\n")
            file.write(f"Postpartum Support: {postpartum_support}\n")
        file.write(f"---------------------------------\n")

    # Redirect to chatbot after saving data
    return render_template('index.html')

# Route for handling chatbot interaction
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    image_url = data.get('imageUrl', '')

    # Process the message and image as needed
    response_message = query_retro_rag(user_message)
    
    return jsonify({'response': response_message})

# Route for handling image uploads
@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'response': "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'response': "No selected file"}), 400

    # Save the file with a secure filename
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Run segmentation on the uploaded image
    output_image_url = run_segmentation(file_path)

    # Return the image URL and the output mask URL to the frontend
    return jsonify({'response': output_image_url})

if __name__ == '__main__':
    app.run(debug=True)
