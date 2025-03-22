import requests
import re
import spacy
import nltk
from bs4 import BeautifulSoup
from newspaper import Article
from transformers import pipeline
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK resources
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load spaCy Named Entity Recognition model
nlp = spacy.load("en_core_web_sm")

# Load BERT Sentiment Analyzer
bert_sentiment = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

# Function to fetch valid news URLs from Google
def get_valid_news_urls(company_name):
    search_url = f'https://www.google.com/search?q={company_name}+news&tbm=nws'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(search_url, headers=headers)

    if response.status_code != 200:
        print("⚠️ Google News request failed!")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    for g in soup.find_all('a', href=True):
        url_match = re.search(r'(https?://\S+)', g['href'])
        if url_match:
            url = url_match.group(1).split('&')[0]
            if "google.com" not in url:  # Ignore Google-related URLs
                links.append(url)
    
    return links[:10]  # Limit to top 10 results

# Function to extract article content using multiple methods
def extract_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"⚠️ Newspaper3k failed: {e}")
    
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code != 200:
            raise Exception("Request failed")
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return '\n'.join(p.text for p in paragraphs if p.text)
    except Exception as e:
        print(f"⚠️ BeautifulSoup failed: {e}")
    
    try:
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(url)
        page_content = driver.page_source
        driver.quit()
        soup = BeautifulSoup(page_content, 'html.parser')
        paragraphs = soup.find_all('p')
        return '\n'.join(p.text for p in paragraphs if p.text)
    except Exception as e:
        print(f"⚠️ Selenium failed: {e}")
    
    return None

# Function to perform Named Entity Recognition (NER) and filter relevant sentences
def filter_relevant_sentences(text, company_name):
    doc = nlp(text)
    relevant_sentences = []

    for sent in text.split('. '):  # Split into sentences
        doc_sent = nlp(sent)
        for ent in doc_sent.ents:
            if company_name.lower() in ent.text.lower():
                relevant_sentences.append(sent)
                break  # Avoid duplicate entries
    
    return '. '.join(relevant_sentences) if relevant_sentences else text  # Return original if no filtering

# Function to perform sentiment analysis using VADER and BERT
def analyze_sentiment(text):
    if not text.strip():
        return "Neutral", 0.0  # Return neutral if text is empty
    
    # VADER Sentiment
    vader_scores = sia.polarity_scores(text)
    vader_compound = vader_scores['compound']
    
    # BERT Sentiment
    bert_result = bert_sentiment(text[:512])[0]  # Limit to 512 tokens
    bert_label = bert_result['label']
    bert_score = bert_result['score']
    
    # Convert BERT result to numerical value
    bert_value = bert_score if bert_label == "POSITIVE" else -bert_score

    # Final sentiment decision (average VADER & BERT)
    final_sentiment = (vader_compound + bert_value) / 2

    # Assign sentiment category
    if final_sentiment > 0.2:
        return "Positive", final_sentiment
    elif final_sentiment < -0.2:
        return "Negative", final_sentiment
    else:
        return "Neutral", final_sentiment

# Main function
def main():
    company_name = input("Enter company name: ")
    print(f"\n🔎 Searching news for: {company_name}\n")
    urls = get_valid_news_urls(company_name)
    
    for i, url in enumerate(urls, 1):
        print(f"\n🔗 Article {i}: {url}\n")
        content = extract_article_content(url)
        
        if content:
            filtered_text = filter_relevant_sentences(content, company_name)
            sentiment, score = analyze_sentiment(filtered_text)
            
            print(f"📰 Extracted Content:\n{filtered_text[:500]}...")  # Show only first 500 characters
            print(f"📊 Sentiment: {sentiment} (Score: {score:.2f})")
        else:
            print("⚠️ Failed to extract content....")

if __name__ == "__main__":
    main()
