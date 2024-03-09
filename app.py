from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

questions = {
    1: "What is the capital of France?",
    2: "Explain the process of photosynthesis.",
    3: "Describe the structure of a cell.",
    4: "What are the causes of global warming?",
    5: "Explain the theory of evolution by natural selection.",
    6: "What are the major functions of the nervous system?",
    7: "Describe the process of DNA replication.",
    8: "What is the role of enzymes in biological reactions?",
    9: "Explain the concept of supply and demand.",
    10: "Describe the impact of globalization on economies."
}

answers = {
    1: "The capital of France is Paris.",
    2: "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll, carbon dioxide, and water. The overall reaction is typically represented as: 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2.",
    3: "Cells are the basic structural and functional units of living organisms. They contain various organelles such as the nucleus, mitochondria, endoplasmic reticulum, Golgi apparatus, and others, each with specific functions.",
    4: "Global warming is primarily caused by the increase in greenhouse gases, such as carbon dioxide (CO2), methane (CH4), and nitrous oxide (N2O), due to human activities such as burning fossil fuels, deforestation, and industrial processes.",
    5: "The theory of evolution by natural selection, proposed by Charles Darwin, states that organisms with traits better suited to their environment are more likely to survive and reproduce, passing on those advantageous traits to future generations. Over time, this process leads to the gradual change of species.",
    6: "The nervous system controls and coordinates body functions and activities. Its major functions include receiving sensory input, processing and interpreting information, and sending appropriate motor responses to effectors such as muscles and glands.",
    7: "DNA replication is the process by which a DNA molecule makes an identical copy of itself. It involves the unwinding of the DNA double helix, separation of the two strands, and the synthesis of new complementary strands using existing strands as templates.",
    8: "Enzymes are biological catalysts that speed up chemical reactions by lowering the activation energy required for the reaction to occur. They facilitate biochemical reactions in living organisms without being consumed in the process.",
    9: "Supply and demand is an economic model that describes the relationship between the availability (supply) and desire (demand) for goods and services. It suggests that the price of a product or service is determined by the balance between supply and demand.",
    10: "Globalization refers to the increasing interconnectedness and interdependence of economies around the world. It has led to greater trade, investment, and cultural exchange, but also to challenges such as income inequality, job displacement, and environmental degradation."
}



@app.route('/')
def index():
    return jsonify({'message': 'Welcome to the Flask backend!'})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'message': 'No file selected for uploading'}), 400

    if file:
        # Here you can perform any processing or save the file as needed
        # For now, just return a success message
        return jsonify({'questions': questions, 'answers': answers}), 200



def calculate_similarity(user_answer, model_answer):
    import random
    return random.randint(0, 100)

@app.route('/calculatesimilarity', methods=['POST'])
def similarity():
    data = request.get_json()
    user_answer = data['userAnswer']
    model_answer = data['modelAnswer']

    # Calculate similarity
    similarity_score = calculate_similarity(user_answer, model_answer)

    # Return the similarity score to the frontend
    return jsonify({'similarityScore': similarity_score})




if __name__ == '__main__':
    app.run(debug=True)
