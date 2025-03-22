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
import time

# Download NLTK resources
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load spaCy Named Entity Recognition model
nlp = spacy.load("en_core_web_sm")

# Load BERT Sentiment Analyzer
bert_sentiment = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

def get_valid_news_urls(company_name):
    search_url = f'https://www.google.com/search?q={company_name}+news&tbm=nws'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"⚠️ Google News request failed: {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    links = set()
    for g in soup.find_all('a', href=True):
        url_match = re.search(r'(https?://\S+)', g['href'])
        if url_match:
            url = url_match.group(1).split('&')[0]
            if "google.com" not in url:  # Ignore Google-related URLs
                links.add(url)
    
    return list(links)[:10]  # Limit to top 10 results

def extract_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        if article.text.strip():
            return article.text
    except Exception as e:
        print(f"⚠️ Newspaper3k failed: {e}")
    
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = '\n'.join(p.text for p in paragraphs if p.text)
        if text.strip():
            return text
    except Exception as e:
        print(f"⚠️ BeautifulSoup failed: {e}")
    
    try:
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(url)
        time.sleep(3)  # Allow time for JavaScript to load content
        page_content = driver.page_source
        driver.quit()
        
        soup = BeautifulSoup(page_content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = '\n'.join(p.text for p in paragraphs if p.text)
        if text.strip():
            return text
    except Exception as e:
        print(f"⚠️ Selenium failed: {e}")
    
    return None

def filter_relevant_sentences(text, company_name):
    doc = nlp(text)
    relevant_sentences = []

    for sent in text.split('. '):
        doc_sent = nlp(sent)
        for ent in doc_sent.ents:
            if company_name.lower() in ent.text.lower():
                relevant_sentences.append(sent)
                break  
    
    return '. '.join(relevant_sentences) if relevant_sentences else text  

def analyze_sentiment(text):
    if not text.strip():
        return "Neutral", 0.0  
    
    vader_scores = sia.polarity_scores(text)
    vader_compound = vader_scores['compound']
    
    try:
        bert_result = bert_sentiment(text[:512])[0]  # Limit to 512 tokens
        bert_label = bert_result['label']
        bert_score = bert_result['score']
        bert_value = bert_score if bert_label == "POSITIVE" else -bert_score
    except Exception as e:
        print(f"⚠️ BERT sentiment analysis failed: {e}")
        bert_value = 0.0  
    
    final_sentiment = (vader_compound + bert_value) / 2
    
    if final_sentiment > 0.2:
        return "Positive", final_sentiment
    elif final_sentiment < -0.2:
        return "Negative", final_sentiment
    else:
        return "Neutral", final_sentiment

def main():
    company_name = input("Enter company name: ")
    print(f"\n🔎 Searching news for: {company_name}\n")
    urls = get_valid_news_urls(company_name)
    
    if not urls:
        print("❌ No valid news URLs found.")
        return
    
    seen_articles = set()
    
    for i, url in enumerate(urls, 1):
        if url in seen_articles:
            continue  
        seen_articles.add(url)
        
        print(f"\n🔗 Article {i}: {url}\n")
        content = extract_article_content(url)
        
        if content:
            filtered_text = filter_relevant_sentences(content, company_name)
            sentiment, score = analyze_sentiment(filtered_text)
            
            print(f"📰 Extracted Content:\n{filtered_text[:500]}...")  
            print(f"📊 Sentiment: {sentiment} (Score: {score:.2f})")
        else:
            print("⚠️ Failed to extract content....")

if __name__ == "__main__":
    main()