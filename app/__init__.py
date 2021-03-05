from flask import Flask, render_template, request, url_for
import os
import pickle
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer

stopwords = stopwords.words('english')

app = Flask(__name__, template_folder='templates')

vectorizer_file = "vectorizer_ngram3.pkl"
classifier_file = "comments_model_ngram3.pkl"
cur_dir = os.path.dirname(__file__)

with open(os.path.join(cur_dir,'models', vectorizer_file), 'rb') as f:
    vectorizer = pickle.load(f)
with open(os.path.join(cur_dir,'models', classifier_file), 'rb') as f:
    model = pickle.load(f)

#####  pipeline from jupyter notebook ######
translator = str.maketrans('', '', string.punctuation)

def remove_stopwords(a):
    return " ".join([word for word in nltk.word_tokenize(a) if word not in stopwords])

def remove_sp_char(a):
    return a.translate(translator)

def text_pipeline2(a):
    a = remove_sp_char(a.lower())
    a = remove_stopwords(a)
    return a
###################################################

history = ["-", "-", "-"]
history_classification = ["-", "-", "-"]
history_score = ["-", "-", "-"]

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    global history
    global history_classification
    global history_score

    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        # Get the input from the user.
        user_input_text = request.form['user_input_text']

        # Turn the text into numbers using our vectorizer
        X = vectorizer.transform([user_input_text])

        # Make a prediction
        predictions = model.predict(X)

        # Get the first and only value of the prediction.
        prediction = predictions[0]

        # Get the predicted probabs
        predicted_probas = model.predict_proba(X)

        # Get the value of the first, and only, predicted proba.
        predicted_proba = predicted_probas[0]

        # The first element in the predicted probabs is the good posts
        bad_comment = predicted_proba[0]

        # The second element in predicted probas is % republican
        good_comment = predicted_proba[1]

        good_p = good_comment.copy()
        bad_p = 1-good_p

        if bad_p > 0.6:
            prediction = "bad"
        elif good_p > 0.6:
            prediction = "good"
        else:
            prediction = "Undecided"

        history.append(user_input_text)
        history_classification.append(prediction)
        history_score.append(str(good_p)[:5])
        if len(history) >  3:
            history = history[1:]
            history_classification = history_classification[1:]
            history_score = history_score[1:]

        return render_template('index.html',
            input_text=user_input_text,
            result=prediction,
            bad_percent=bad_p*100,
            good_percent=good_p*100,
            r11 = history[2],
            r12 = history_score[2],
            r13 = history_classification[2],
            r21 = history[1],
            r22 = history_score[1],
            r23 = history_classification[1],
            r31 = history[0],
            r32 = history_score[0],
            r33 = history_classification[0]
            )

@app.route('/bootstrap_twitter/')
def bootstrap():
    return render_template('bootstrap_twitter.html')

@app.route('/about_us/')
def about_us():
    return render_template('about_us.html')

if __name__ == '__main__':
    app.run(debug=True)