import tweepy
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords


# Twitter API Anahtarları
consumer_key = " "
consumer_secret = ""
access_token = ""
access_token_secret = ""

# Tweepy ile Twitter API'ye bağlanma
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Temel NLP için NLTK'nin önişleme adımları
nltk.download('stopwords')
stop_words = set(stopwords.words('turkish'))

def preprocess_tweet(tweet):
    # Tweet'i temizleme: kullanıcı adları, linkler ve özel karakterler
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)
    tweet = re.sub('https?://[A-Za-z0-9./]+', '', tweet)
    tweet = re.sub("[^a-zA-Z]", ' ', tweet)

    # Küçük harfe dönüştürme
    tweet = tweet.lower()

    # Stop-words'leri kaldırma
    tweet_tokens = nltk.word_tokenize(tweet)
    tweet_tokens = [word for word in tweet_tokens if word not in stop_words]

    return ' '.join(tweet_tokens)

# Belirli bir konu üzerindeki tweet'leri çekme
topic = "your topic"
tweet_count = 10  
tweets = tweepy.Cursor(api.search_tweets, tweet_mode='extended',q=topic, lang="tr").items(tweet_count)


for tweet in tweets:
    # Tweet metnini al
    tweet_text = tweet.text

    # Temel ön işleme
    processed_tweet = preprocess_tweet(tweet_text)

    # İşlenmiş tweet'i ekrana yazdır
    print(processed_tweet)

    # Duygu analizi örneği (TextBlob kullanarak)
    blob = TextBlob(processed_tweet)
    sentiment = blob.sentiment
    print("Duygu Analizi:", sentiment)
    print("------")