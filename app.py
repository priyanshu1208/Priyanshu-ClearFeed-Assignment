from flask import Flask, jsonify, request, render_template
from Technique_1_bm250 import evaluate_bm25
from Technique_2_tfIdf_Cosine import evaluate_cosine
from Technique_3_sentence_transformer  import evaluate_sent
from Technique_4_GoogleRequest import main
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query_api():
    
    data = request.json
    query = data.get('query_question', '')
    model_number = int(data.get('model_number', 1))
    
    link =[]
    answer=""
    json_file_path = 'Clearfeed_kb.json'

    with open(json_file_path, 'r') as file:
        documentation_json = json.load(file)
    if model_number==1:
        link,answer = evaluate_bm25(query,documentation_json)
        print(link)
    elif model_number ==2:
        link,answer = evaluate_cosine(query,documentation_json)
    elif model_number==3:
        link,answer = evaluate_sent(query,documentation_json)
    elif model_number ==4:
        link,answer = main(query,documentation_json)



    return jsonify({'links': link, 'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
