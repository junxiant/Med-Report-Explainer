# app.py
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import PyPDF2
from openai import OpenAI
import re

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure OpenAI client to use local LM Studio server
client = OpenAI(base_url="http://172.30.240.1:1234/v1", api_key="lm-studio")

# You may need to replace this with the actual identifier of your model
MODEL_IDENTIFIER = "model-identifier"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def explain_medical_terms(text):
    try:
        completion = client.chat.completions.create(
            model=MODEL_IDENTIFIER,
            messages=[
                {"role": "system", "content": "You are a medical expert. Explain medical terms and jargon for patients to understand. \
                    Make it short, clear and concise. \
                    For each term or phrase, provide the explanation in this format: \
                    'Term: [term or phrase]\n\nExplanation: [simple explanation]'\n\n \
                    Avoid any speculation or creativity. \
                    All of the explanation must be factual. \
                    Just output the explanations, don't use an"},
                {"role": "user", "content": f"Explain the medical terms and jargon in the following text so the patient can understand it better:\n\n{text}"}
            ],
            temperature=0.25,
        )
        explanations = completion.choices[0].message.content.strip()
        
        # Remove any introductory or concluding text
        explanations = re.sub(r'^.*?(?=Term:)', '', explanations, flags=re.DOTALL)
        explanations = re.sub(r'\n\s*$', '', explanations)
        
        # Parse the explanations into a structured format
        parsed_explanations = []
        for explanation in re.split(r'\n(?=Term:)', explanations):
            parts = explanation.split('\n', 1)
            if len(parts) == 2:
                term = parts[0].replace('Term:', '').strip()
                explanation = parts[1].replace('Explanation:', '').strip()
                parsed_explanations.append({"term": term, "explanation": explanation})
        
        return parsed_explanations
    
    except Exception as e:
        print(f"Error communicating with LLM server: {e}")
        return [{"term": "Error", "explanation": f"Unable to generate explanation. Please try again later. Details: {str(e)}"}]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files and 'text' not in request.form:
            return jsonify({'error': 'No file or text provided'}), 400
        
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                if file_path.endswith('.pdf'):
                    text = extract_text_from_pdf(file_path)
                else:
                    with open(file_path, 'r') as f:
                        text = f.read()
        else:
            text = request.form['text']
        
        explanations = explain_medical_terms(text)
        return jsonify({'text': text, 'explanations': explanations})
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    
    
