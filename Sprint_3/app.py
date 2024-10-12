from flask import Flask, request, render_template
import pickle

import pandas

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input')
def home():
    return render_template('input_form.html')

@app.route('/result', methods=['POST'])
def result():
    # Collect and validate the input
    try:
        age = float(request.form.get('age'))
        sex = float(request.form.get('sex'))
        query_hyperthyroid = float(request.form.get('query_hyperthyroid'))
        tsh = float(request.form.get('tsh'))
        t3 = float(request.form.get('t3'))
        tt4 = float(request.form.get('tt4'))
        t4u = float(request.form.get('t4u'))
        fti = float(request.form.get('fti'))

        input_features = [[age, sex, query_hyperthyroid, tsh, t3, tt4, t4u, fti]]

        # Load the model and make a prediction
        with open('rf_Grid.pkl', 'rb') as model_file:
            rf = pickle.load(model_file)
        input_features=pandas.DataFrame(input_features, columns=['age', 'sex', 'query_hyperthyroid', 'TSH', 'T3', 'TT4', 'T4U', 'FTI'])
        prediction = rf.predict(input_features)
        result = prediction[0]

        return render_template('result.html', res=result)
    except Exception as e:
        # Handle exceptions (e.g., missing file, invalid input)
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
