from flask import Flask,render_template,request
import joblib

# initializing the app
app = Flask(__name__)

# loading the model

tfidf = joblib.load('tfidf_model.pkl')
model = joblib.load('netflix_nlp.pkl')


@app.route('/',)
def hello():
    return render_template('form.html')


@app.route('/submit' , methods = ["POST"])
def submit():
    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vector = tfidf.transform(data).toarray()
        my_pred = model.predict(vector)

    return render_template ('predict.html', prediction = my_pred)    


if __name__ =='__main__':
    app.run(debug=True)
