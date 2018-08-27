# To build model / run API:

(A virtual python 3.6 environment is recommended.)

	$ pip install -r requirements.txt

(Using pre trained model and preprocessed data:)
	$ python -W ignore mk.py 

OR 

(Preprocess data and train:)
	$ python -W ignore mk.py --preprocess_data --do_train

# To make API calls:

$ curl  -H "Content-Type: applicatiline": "Parenting Lessons From a Partial Eclipse", "summary": "summary_content", "worker_id": 1 }' http://localhost:8080

# Please provide a brief writeup of your approach and decision making process regarding subjective design choices you made during the analysis, along with instructions for executing the program if necessary.

This was a fun and tricky challenge. Notably:
- not many features
- multi-label classification is harder than multi-class classification

For efficiency of time, I went with decision tree classifier model -- however, I would be interested to try to implement an LSTM, developing a more coherent "hidden layer" representation of our words (as opposed to my weighted average of word embeddings) would, I think, perform better. Maybe? Additionally, neural net models seem well suited to such multi-label output, what with their softmax architecture. 

Again, I made a brute decision to just process the summary text, as I assumed that this would hold more semantic information. It was generally longer, and also titles can sometimes be cute, whereas it seems to be that the summary holds more clear representation of article content, and hence, emotional evocation.

I also added the `worker_id`.  Perhaps this is the id of the worker who does the emotional labelling. One might think that different workers are predisposed to rate according to a common patter (e.g. I am more inclined ot think an article is "fearsome" than you, you are more inclined to think it is "warm and loving".) There are only 1306 raters and 90K articles, so with more time I would investigate this potential a bit more, but at present it is indeed included in features.  

I opted for this more sophisticated approach after seeing little success with my held out tests using just simple TIDF matrices as inputs. In the end, my more complicated model of TFIDF weighted word vectors didn't perform better, with Hamming loss scores around .15. This is really quite bad, as I see it. We have 10 labels, with 0-8 often 0, and label 9 often 1. In `eda.py` I explore the data, and see that for most articles, roughly 2 emotions are flagged as 1. So, in some naive sense, if we predict [0,0,0,0,0,0,0,0,0,1] each time, it seems that we should often only be making 1 mistake out of 10 (missing the addition 1 for one of the variables 0-8). This would be a Hamming loss of .1, so my model is not promising at all.

That said, I'm optimistic this architecture could be massaged, or the model could be tweaked (hyper parameters?) to get better results.

Lastly, there is always the real chance that there's a mistake in my code, which would also explain these poor results. Such is the power of collaboration. I'd appreciate any feedback on my model, and a discussion of how you would / are already going about this at the NYT.
