import os
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import base64
import json
from image_processing import process_image  # Importing the image processing function
from video_processing import process_video  # Importing the video processing function

app = Flask(__name__)

# Define allowed file extensions for images and videos
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'}

# Function to check if the file has a valid extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Default route (home page)
@app.route('/')
def home():
    return render_template('index.html')  # This will render the index.html page

# Upload file endpoint
@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()

        try:
            if not os.path.exists("uploads"):
                os.makedirs("uploads")

            file_path = os.path.join("uploads", filename)
            file.save(file_path)

            # Handling image files
            if file_extension in ['png', 'jpg', 'jpeg', 'gif']:
                processed_data = json.loads(process_image(file_path))

                return jsonify({
                    "relationships": processed_data.get("relationships", []),
                    "original_image": processed_data.get("original_image"),
                    "image": processed_data.get("image")
                }), 200

            # Handling video files
            elif file_extension in ['mp4', 'avi']:
                processed_data = json.loads(process_video(file_path))
                video_url = f"/uploads/{filename}"

                return jsonify({
                    "relationships": processed_data.get("relationships", []),
                    "video_url": video_url,
                    "video_frames": processed_data.get("video_frames", [])
                }), 200

        except Exception as e:
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500

    return jsonify({"error": "File type not allowed"}), 400

# Serve uploaded files
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_file(os.path.join("uploads", filename))

# Favicon route (to prevent 404 for /favicon.ico requests)
@app.route('/favicon.ico')
def favicon():
    return '', 204  # Empty response for favicon requests

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)