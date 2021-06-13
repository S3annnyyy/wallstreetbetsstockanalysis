import string
import re
import time
import sys
import nltk
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from ast import literal_eval
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords, wordnet as wn
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
nltk.download('wordnet')

# Update lexicon for better accuracy
positive_words = 'buy bull long support undervalued underpriced cheap upward rising trend moon rocket hold hodl breakout call beat support buying holding high profit stonks yolo'
negative_words = 'sell bear bubble bearish short overvalued overbought overpriced expensive downward falling sold sell low put miss resistance squeeze cover seller loss '
pos = {i: 5 for i in positive_words.split(" ")}
neg = {i: -5 for i in negative_words.split(" ")}
stock_lexicons = {**pos, **neg}
analyser = SentimentIntensityAnalyzer()
analyser.lexicon.update(stock_lexicons)

# configure NER
nlp = spacy.load("en_core_web_trf")


def main():
    # check command line argument
    if len(sys.argv) != 2:
        sys.exit("Usage: EDA.py dataset")

    # time
    start_time = time.time()

    # get dataset
    corpus = pd.read_csv(r'C:\\Users\Sean\PycharmProjects\wallstreetbetsanalysis\{}'.format(sys.argv[1]), encoding="latin-1")
    corpus.dropna(inplace=True)
    corpus.reset_index(drop=True, inplace=True)

    # Text preprocessing
    corpus["header"] = corpus["header"].apply(preprocess_text)

    # Exploratory Data Analysis

    # Volume of posts across months
    # convert unix timestamp to Month Year
    corpus["date"] = corpus["utc"].apply(utctodatetime)
    # gather data
    vol, month_year = volume_by_monthyear(corpus["date"])
    # visualise
    plt.bar(month_year, vol, color="sandybrown", edgecolor="black")
    plt.gca().spines["top"].set_alpha(0.0)
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.0)
    plt.gca().spines["left"].set_alpha(0.3)
    plt.title('Number of posts across the months', fontsize=16, ha="center")
    plt.ylabel("Number of posts")
    plt.xlabel('Month')
    plt.suptitle('r/wallstreetbets', fontsize=24, y=0.95, ha="center")
    plt.show()

    # Sentiment polarity score distribution
    corpus["text"] = corpus["text"].astype(str)
    for index, sentence in tqdm(enumerate(corpus["text"])):
        sentiment_rating = sentiment_analyzer(sentence)
        corpus.loc[index, "polarity"] = float(sentiment_rating)
    plt.gca().spines["top"].set_alpha(0.0)
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.0)
    plt.gca().spines["left"].set_alpha(0.3)
    plt.hist(corpus["polarity"], color="sandybrown", edgecolor="black", bins=20)
    plt.suptitle('r/wallstreetbets', fontsize=24, y=0.95, ha="center")
    plt.title("Sentiment Polarity Distribution", fontsize=16, ha="center")
    plt.ylabel('Number of posts')
    plt.xlabel('Sentiment score')
    plt.show()

    # Post word count distribution
    corpus["text length"] = corpus["text"].apply(word_counter)
    plt.scatter(corpus.index, corpus["text length"], c=corpus["text length"], cmap="BuPu_r")
    plt.gca().spines["top"].set_alpha(0.0)
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.0)
    plt.gca().spines["left"].set_alpha(0.3)
    plt.suptitle('r/wallstreetbets', fontsize=24, y=0.95, ha="center")
    plt.title('Word count distribution of posts')
    plt.ylabel('Word Count')
    plt.xlabel("Posts index")
    plt.show()

    # Topic modelling using LDA
    n_features = 250
    n_components = 10
    n_topwords = 10
    # Text to vectors
    vectorizer = TfidfVectorizer(max_features=n_features)
    vectorizer_matrix = vectorizer.fit_transform(corpus["header"])
    lda_model = LatentDirichletAllocation(n_components=n_components, max_iter=5, random_state=0, learning_method='online')
    lda_model.fit(vectorizer_matrix)
    feature_names = vectorizer.get_feature_names()

    # Visualize
    plot_lda_model(lda_model, feature_names, n_topwords, "Topics in LDA Model")

    # NER
    corpus["organizations"] = name_entity_recognition(corpus)
    print(corpus["organizations"])
    # export dataframe into csv file
    corpus.to_csv(r"C:\\Users\Sean\PycharmProjects\wallstreetbetsanalysis\datasetlarge2.csv")

    # sort
    blacklist = ["WSB", "Robinhood", "SEC", "Fed", "CNBC", "Citadel", "RH", "FDA", "Fidelity", "Reddit"]
    ticker_data = get_ticker_frequency(corpus)
    # visualize
    y = [v[0] for v in ticker_data if v[1] not in blacklist][:20]
    x = [v[1] for v in ticker_data if v[1] not in blacklist][:20]
    plt.bar(x, y, color="sandybrown", edgecolor="black")
    plt.title("20 most mentioned stocks")
    plt.xlabel("Companies")
    plt.ylabel("No. of mentions")
    plt.show()

    # export dataframe into csv file
    corpus.to_csv(r"C:\\Users\Sean\PycharmProjects\wallstreetbetsanalysis\datasetlargeV3.csv", index=False)
    print("{}mins|{}seconds".format(round(((time.time() - start_time) / 60), 1), round((time.time() - start_time))))


def utctodatetime(utc):
    date = datetime.strftime(datetime.fromtimestamp(utc), "%b %Y")
    return date


def volume_by_monthyear(date):
    volume = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    month_year = []
    index = -1
    for j in range(len(date)):
        if date[j] not in month_year:
            index += 1
            month_year.append(date[j])
            volume[index] += 1
        else:
            volume[index] += 1
    return volume, month_year


def get_wordnet_posttag(word):
    """This function takes in a word which then allocates a tag. Eg: "fish" gets a tag v which is VERB"""
    # added in [0][1][0] so as to get main letter. eg VBG, "approaching" outcome ->[('approaching', 'VBG')]
    # [0][1][0] gets "V" only
    tagged = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wn.ADJ,
        "N": wn.NOUN,
        "V": wn.VERB,
        "R": wn.ADV
    }
    return tag_dict.get(tagged, wn.NOUN)  # The OG description is always NOUN


def preprocess_text(text):
    lemmatize_the_word = WordNetLemmatizer()
    replace_annotations = re.compile('[/(){}\[\]\|@,;]')
    replace_symbols = re.compile('[^0-9a-z #+_]')
    # set text to lowercase
    text = text.lower()
    # remove any bad symbols
    text = replace_annotations.sub(' ', text)
    text = replace_symbols.sub('', text)
    text = " ".join(word for word in text.split() if word not in stopwords.words('english') and word.isalpha())
    product = []
    for word in text.split():
        tag = get_wordnet_posttag(word)
        if word not in stopwords.words('english') and word.isalpha():
            lemmatized_word = lemmatize_the_word.lemmatize(word, tag)
            product.append(lemmatized_word)
    text = " ".join(word for word in product)
    return text


def plot_lda_model(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7, color="sandybrown")
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 25})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=25)

    plt.subplots_adjust(top=0.9, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


def sentiment_analyzer(text):
    score = analyser.polarity_scores(text)
    return score['compound']


def word_counter(sentence):
    count = sum([word.strip(string.punctuation).isalpha() for word in sentence.split()])
    return count


def name_entity_recognition(corpus):
    """This function uses NER to get words defined as ORG under post headers. If none, it will proceed
    to post text. Output will be a list containing ORGs and if no ORGs found, function returns "No label" """
    def get_entity(col):
        temp = nlp(col)
        org = [entity for entity in temp.ents if entity.label_ == "ORG"]
        print(org)
        return list(set(org))

    x = corpus["header"].apply(get_entity)
    y = corpus["text"].apply(get_entity)
    merged = x + y
    return merged


def get_ticker_frequency(df):
    """This function takes in a corpus, which then filters its tickers from organizations column. Sorts and count
    frequency of each ticker and returns top 10 mentioned tickers"""
    def tickerfreq(wordlist):
        wordfreq = [wordlist.count(j) for j in wordlist]
        return dict(set(zip(wordlist, wordfreq)))

    def sortfreq(x):
        temp = [(x[key], key) for key in x]
        temp.sort()
        temp.reverse()
        return temp

    tickerlist = []
    for index, row in df.iterrows():
        if row["organizations"] == "[]":  # eliminate rows which NER did not capture
            continue
        else:
            tickers = row["organizations"].strip('[]').split(', ')
            for i in tickers:
                tickerlist.append(i)
    return sortfreq(tickerfreq(tickerlist))


if __name__ == "__main__":
    main()
