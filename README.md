# wqd7006-ML-proj
This is a project repository for subject id WQD7006

# Title: How Sarcastic Are You?

1.0	Introduction

  There are several different emerging area in sentiment analysis. It is an analysis that analyze people perception, attitude and opinion. The main objective of sentiment analysis is to determine the polarity of a given sentence, such as positive, negative and neutral. Sarcasm is another branch in sentiment analysis which didn’t follow the exact polarity of a given text. Sarcasm is a form of irony that uses language which usually contradict with the intended meaning. In a face-to-face conversation, the person’s body language, vocal intonation, facial expression, etc. could help in identifying sarcasm. However, when it related to social media and other text-based materials, sarcasm detection become more complicated. To classify sarcasm in text, there are certain cues and labelling need to be attached in order to label them properly.
	
  The reason researchers nowadays interested in detecting sarcasm online, are to improve in interpreting customer sentiment, identify public perception of certain product, understanding political commentary and predicting social media communication information on certain topic. There are two approaches used to detect sarcastic sentences, rule-based and classical machine learning and deep learning approaches. In the first method (rule-based approaches), the sarcastic sentence will be recognize using specific evidence like positive verb, negative word, etc. as mentioned in Riloff et al. (2013). In the second method (machine learning and deep learning-based approaches), mostly researchers used bag of words as their features to classify sarcasm. Like in Khodak et al. (2019), the author employed bag-of-word to convert unstructured data to structured data.

objectives:
	
1. To determine the minimum set of features required for sarcasm detection.
2. To identify if the sentences is sarcastic or non sarcastic by using the best train models.
3. to test the final model with sample of sentences provided.
	
  
	The contribution of this study...

2.0	Methodology

	2.1.	Datasets

	•News Headlines Dataset: This dataset can be download from “Kaggle” website. It consists of 26,709 news headlines collected from the “Onion” and “HuffPost” websites. Among the 26,709, 14,985 headlines labeled as “Not sarcastic” and 11,724 headlines are labeled as “Sarcastic”. This dataset will provide the labeled data for our study. 

	•Friends Script Dataset: The TV show F.R.I.E.N.D.S transcripts were downloaded from https://fangj.github.io/friends/ where each scene begins with a description of the location and situation followed by series of utterances spoken by characters. The reason behind choosing TV show transcript as our dataset was to restrict to a small set of characters that use a lot of humor. These characters are often sarcastic towards each other because of their strong inter-personal relationship. 

2.2.	Data Cleaning

2.3.	Data Preprocessing

2.4.	Features Extraction

2.5.	Machine Learning

	2.5.1.	Linear Support Vector Machine (SVM)
~~
	2.5.2.	Naïve Bayes 
~~
	2.5.3.	Logistic Regression
~~
	2.5.4.	Random Forest Regression
~~
	2.5.5.	Long-Short-Term-Memory (LSTM)
~~
3.0	Results

	3.1.	Model Evaluation

4.0	Discussion

5.0	Conclusion

6.0	References

