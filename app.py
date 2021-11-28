from flask import Flask,request,jsonify,render_template
import numpy as np
import pickle

app=Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('transform.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    final_transformed_features = scaler.transform(final_features)
    prediction = model.predict(final_transformed_features)
    
    
    if prediction==1:
        result="Diabetic"
    else:
        result="Not Diabetic"
    
    return render_template('index.html',prediction='Person is: {}' .format(result))
    

if __name__=='__main__':
    app.run(debug=True)