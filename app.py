from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from rag import main as rag_main
from similarity import calculate_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        if page.extract_text() is not None:
            text += page.extract_text()
    return text

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'message': 'No file part in the request'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'message': 'No file selected for uploading'}), 400

        if file:
            raw_text = get_pdf_text(file)
            questionNo = request.form.get('questionNo')
            # Call the main function from rag.py with the raw text as parameter
            questions, answers = rag_main(raw_text,questionNo)
            return jsonify({'questions': questions, 'answers': answers}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/calculatesimilarity', methods=['POST'])
def similarity():
    try:
        data = request.get_json()
        if 'userAnswer' not in data or 'modelAnswer' not in data:
            return jsonify({'error': 'Missing userAnswer or modelAnswer in request'}), 400
        user_answer = data['userAnswer']
        model_answer = data['modelAnswer']

        # Calculate similarity
        similarity_score = calculate_similarity(user_answer, model_answer)

        # Return the similarity score to the frontend
        return jsonify({'similarityScore': similarity_score})
    except Exception as e:
        return jsonify({'error': 'Failed to calculate similarity', 'details': str(e)}), 500

if __name__ == '__main__':
 app.run(debug=True)