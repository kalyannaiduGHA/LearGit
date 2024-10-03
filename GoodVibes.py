#import necessary libraries

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Gather your data

reviews = ['This movie was fantastic! A must-watch.',
           'I didn\'t enjoy this movie at all.',
           'Amazing storyline and great acting!',
           'The plot was dull and predictable.',
           'Loved the cinematography! Highly recommended.']

#Label your data, Label each review as either 'Positive' or 'Negative'

labels = ['positive', 'negative', 'positive', 'negative', 'positive']

#Vectorize the TextData, Convert your text data into numbers that the computer can understand using CountVectorixer

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(reviews)

#split the data, Split the data into training and testing sets so the computer can learn from some data and be t4ested on the rest

x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

#Train the Model, Create a naive Bayes classisifer and train it using the training data
model = MultinomialNB()
model.fit(x_train, y_train)

#Test the Model, Use the trained model to predict the vibes of the test data
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)

#interpret the results, Finally you can decide if the movie has Good Vibes based on the accuracy of your model. If the accuracy is above 80%,print "Good VIbes"

if accuracy > 0.8:
    print('Good vibes. Book the ticket!')
else:
    print('Needs more work!')