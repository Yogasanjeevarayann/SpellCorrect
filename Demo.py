from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('city_correction_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to correct spelling
def correct_spelling(input_city):
    input_vector = vectorizer.transform([input_city])
    corrected_city = model.predict(input_vector)[0]
    return corrected_city   

# API endpoint for correcting city spelling
@app.route('/api/correct_city', methods=['POST'])
def correct_city():
    try:
        data = request.json
        text = data['text']

        # Correct the spelling
        corrected_city = correct_spelling(text)

        return jsonify({'corrected_city': corrected_city})

    except KeyError as e:
        return jsonify({'error': f'Missing key in JSON: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True)
