import re
from collections import defaultdict
import contractions
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk import bigrams
from nltk import trigrams
import pprint

#Get Data
dat_info = pd.read_csv('metacritic_game_info.csv')
dat_review = pd.read_csv('metacritic_game_user_comments.csv')
#Filtering for just the Legend of Zelda: Breath of the Wild
dat_review = dat_review.loc[(dat_review.Title == "The Legend of Zelda: Breath of the Wild")]

#Cleaning the data
dat_info.fillna('', inplace=True)
dat_review.fillna('', inplace=True)

#Creating function to categorize the reviews into Excellent and Bad categories.
def new_label(x):
    if x>7:
        return 'Excellent'
    else:
        return 'Bad'

#Adding the new label to the desired dataframe
dat_review['label'] = dat_review.Userscore.apply(new_label)
#Creating a dataframe of the user scores to visualize the distribution
number_of_reviews = pd.DataFrame(dat_review.Userscore.value_counts())
number_of_reviews.sort_index(inplace=True)
number_of_reviews.columns = ['Number of Reviews']
colors = ['green' if (x>500) else 'lightgreen' for x in number_of_reviews['Number of Reviews']]
sns.countplot(data = dat_review, x = 'Userscore', palette = colors)
plt.show()
#Printing the info about the reviews and also printing the average user score and the critics score given by Metacritic.
print(number_of_reviews.T)
print('Average user score: ', dat_info['Avg_Userscore'][11])
print('Metacritic score: ', dat_info['Metascore'][11])
print()

#Preprocessing text
wnl = nltk.WordNetLemmatizer()
vectorizer = CountVectorizer(analyzer = "word", stop_words = 'english')
stop_words = set(stopwords.words("english"))
dat_review['Comment'] = dat_review['Comment'].apply(lambda comments: [contractions.fix(word) for word in comments.split()])#fix contractions
dat_review['Comment'] = [' '.join(map(str, l)) for l in dat_review['Comment']]#join back into string
dat_review['Comment'] = dat_review['Comment'].str.lower()#convert to lowercase
dat_review['Comment'] = dat_review['Comment'].apply(lambda x: re.sub(r'[^\w\d\s\']+', '', x)) #remove special characters
dat_review['Comment'] = dat_review['Comment'].apply(lambda x: wnl.lemmatize(x)) #Lemmatize
dat_review['Comment'] = dat_review['Comment'].apply(word_tokenize)#tokenize the words
dat_review['Comment'] = dat_review['Comment'].apply(lambda x: [word for word in x if word not in stop_words])#remove stop words
dat_review['Comment'] = [' '.join(map(str, l)) for l in dat_review['Comment']]#join back into string
tokenized = dat_review['Comment'].apply(word_tokenize)

print(dat_review['Comment'].sample(10))

#Displaying wordclouds for each score in the review system
fig = plt.figure(figsize=(15,7))
for i in range(11):
    plt.subplot(4,3,i+1)
    plt.title(f'Score: {i}', loc='center')
    plt.axis('off')
    text = dat_review[dat_review.Userscore==i]
    #Adding additional game specific stop words in order to make the word clouds more meaningful. Game and Zelda were in all of them.
    wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="white", stopwords=set(['game', 'zelda'])).generate(" ".join(text.Comment))
    plt.imshow(wordcloud, interpolation="bilinear")
plt.show()

#Displaying wordclouds for "Bad and Good" Reviews
Good = dat_review[dat_review.label == "Excellent"]
Bad = dat_review[dat_review.label == "Bad"]

wordcloud_good = WordCloud(max_font_size=50, max_words=150, background_color="white", stopwords=set(['game', 'zelda', 'good'])).generate(' '.join(Good.Comment))
plt.title('Good Reviews WordCloud')
plt.imshow(wordcloud_good, interpolation='bilinear')
plt.axis('off')
plt.show()

wordcloud_bad = WordCloud(max_font_size=50, max_words=150, background_color="black", stopwords=set(['game', 'zelda', 'good'])).generate(' '.join(Bad.Comment))
plt.title("Bad Reviews WordCloud")
plt.imshow(wordcloud_bad, interpolation='bilinear')
plt.axis('off')
plt.show()

#Creating function to make bigrams and trigrams
def ngram(x, n):
    """
    Input DataFrame column and either 2 or 3 for a bigram or trigram
    :return: top 10 items of that ngram from that text
    """
    counts_dict = defaultdict(int)
    sent = x
    sentences = sent_tokenize=(sent)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    #Bigrams
    if n == 2:
        for sentence in tokenized_sentences:
            for bigram in bigrams(sentence, pad_right = True, pad_left = True):
                counts_dict[bigram] += 1
    elif n == 3:
        for sentence in tokenized_sentences:
            for trigram in trigrams(sentence, pad_right = True, pad_left = True):
                counts_dict[trigram] += 1
    else:
        print("Please select between a bigram and a trigram")
    ngram_list = [(ngram, counts_dict[ngram]) for ngram in counts_dict.keys()]
    ngrams_sorted = sorted(ngram_list, key = lambda freq: freq[1], reverse=True)
    top_10_ngrams = ngrams_sorted[:10]
    return top_10_ngrams

#Good review Bigrams and Trigrams
Good_Bigrams = ngram(Good.Comment, 2)
Good_Trigrams = ngram(Good.Comment, 3)

#Bad Review Bigrams and Trigrams
Bad_Bigrams = ngram(Bad.Comment, 2)
Bad_Trigrams = ngram(Bad.Comment, 3)

def visualizeNGrams(p, Title, ngram):
    two_tuples = list(zip(*p))
    x = [" ".join([str(i or 'None') for i in ngram_tuple]) for ngram_tuple in two_tuples[0]]
    y = list(two_tuples[1])
    plt.figure(figsize=(15,15))
    plt.xlabel(Title)
    if ngram == 2:
        plt.ylabel("Top 10 Bi-grams")
    elif ngram == 3:
        plt.ylabel("Top 10 Tri-grams")
    else:
        "Please specify a bigram or a trigram"
    sns.barplot(x=y,y=x)
    plt.show()

visualizeNGrams(Good_Bigrams, "Good Bi-gram Frequency", 2)
visualizeNGrams(Good_Trigrams, "Good Tri-gram Frequency", 3)
visualizeNGrams(Bad_Bigrams, "Bad Bi-gram Frequency", 2)
visualizeNGrams(Bad_Trigrams, "Bad Tri-Gram Frequency", 3)