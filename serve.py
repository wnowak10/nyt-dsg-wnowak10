# -*- coding: utf-8 -*-
""" serve.py

    To run the server (using Python 3; there's a note below on Python versions):
        $ python serve.py
    To make a sample request from bash:
        $ 
        curl -d '{ "headline": "headline_content", "summary": "summary_content", "worker_id": 1 }' http://localhost:8080

return { "emotion_0": 0.0, "emotion_1": 0.12, ... "emotion_9": 0.99 }

    A note about Python versions:
        Please run this script using Python 3. You can check your python version
        by running `python -V`. On some systems, you can run python 3 by simply
        typing `python`, on others, `python3` will work. In some cases Python 3
        must be installed.
        
"""
import cgi
import json
import pickle
import spacy

import numpy as np

from http.server              import BaseHTTPRequestHandler, HTTPServer
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils          import sparse2full

# _______________________________________________________________________
# Load `en_core_web_md`...takes a few seconds.
print('Please wait...loading spacy `en_core_web_md`.')
global nlp
nlp = spacy.load('en_core_web_md')

# _______________________________________________________________________
# Server classes
def keep_token(t):
            return (t.is_alpha and 
                not (t.is_space or t.is_punct or 
                     t.is_stop or t.like_num))

def lemmatize_doc(doc):
    return [ t.lemma_ for t in doc if keep_token(t)]

class handler(BaseHTTPRequestHandler):
    """ Return a JSON object detailing emotional state predictions.
    """
    model = None

    def do_POST(self):

        ctype, pdict = cgi.parse_header(self.headers.get('content-type'))

        length = int(self.headers.get('content-length'))
        decoded_string = self.rfile.read(length).decode('utf-8')

        # Extract input to predict. This comes from the user's POST request.
        # Right now, model only uses one of these as input.
        headline = list(json.loads(decoded_string).values())[0]
        summary_content = list(json.loads(decoded_string).values())[1]
        worker_id = list(json.loads(decoded_string).values())[2]
        worker_id = int(worker_id)

        # Choose to use summary_content or headline. 
        docs = [lemmatize_doc(nlp(summary_content))]
        
        docs_corpus = [self.docs_dict.doc2bow(doc) for doc in docs]

        # Use `model_tfidf` trained on training corpus.
        docs_tfidf  = self.model_tfidf[docs_corpus]
        docs_vecs   = np.vstack([sparse2full(c, len(self.docs_dict)) for c in docs_tfidf])

        # self.tfidf_emb_vecs is a matrix of num words in corpus x wod embed dimension.
        docs_emb = np.dot(docs_vecs, self.tfidf_emb_vecs) 

        print(docs_emb.shape)
        # Add worker id to array.
        wid_arry = np.asarray([worker_id]).reshape(1,1)
        X = np.concatenate((docs_emb, wid_arry), axis = 1)
        print(X.shape)
        preds = self.model.predict(X)
        preds = preds[0]

        self.wfile.write(json.dumps({ "emotion_0": preds[0], 
            "emotion_1": preds[1], 
            "emotion_2": preds[2], 
            "emotion_3": preds[3], 
            "emotion_4": preds[4], 
            "emotion_5": preds[5], 
            "emotion_6": preds[6], 
            "emotion_7": preds[7], 
            "emotion_8": preds[8], 
            "emotion_9": preds[9] }).encode('utf-8'))
        print('Ping! Returned predictions to user.')


def run_server(model, docs_dict, model_tfidf, tfidf_emb_vecs):
    print('Starting server...')
    server_address = ('127.0.0.1', 8080)
    handler.model = model
    handler.docs_dict = docs_dict
    handler.model_tfidf = model_tfidf
    handler.tfidf_emb_vecs = tfidf_emb_vecs
    try:
        httpd = HTTPServer(server_address, handler)
        print('Running server...')
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('Shutting down.')


# _______________________________________________________________________
# Bundle to start

def main(model, docs_dict, model_tfidf, tfidf_emb_vecs):
    """
    Load the decision classifier model and the `docs_dict` to build
    a tfidf.
    """
    run_server(model, docs_dict, model_tfidf, tfidf_emb_vecs)

if __name__ == '__main__':
    with open('default_model.pkl', 'rb') as f:
        model = pickle.load(f)

    x_train = pickle.load(open("x_train.pkl", "rb"))
    y_train = pickle.load(open("y_train.pkl", "rb"))
    x_test  = pickle.load(open("x_test.pkl", "rb"))
    y_test  = pickle.load(open("y_test.pkl", "rb"))

    docs_dict      = pickle.load(open("docs_dict.pkl", "rb"))
    model_tfidf    = pickle.load(open("model_tfidf.pkl", "rb"))
    tfidf_emb_vecs = pickle.load(open("tfidf_emb_vecs.pkl", "rb"))
 
    run_server(model, docs_dict, model_tfidf, tfidf_emb_vecs)
