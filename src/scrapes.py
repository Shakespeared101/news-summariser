import requests
import re
from bs4 import BeautifulSoup
from newspaper import Article
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

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
            if "google.com" not in url:
                links.append(url)
    
    return links[:10]  # Limit to top 10 results

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

def main():
    company_name = input("Enter company name: ")
    print(f"\n🔎 Searching news for: {company_name}\n")
    urls = get_valid_news_urls(company_name)
    
    for i, url in enumerate(urls, 1):
        print(f"\n🔗 Article {i}: {url}\n")
        content = extract_article_content(url)
        if content:
            print("📰 Extracted Content:\n", content[:], "...")  # Print first 500 chars
        else:
            print("⚠️ Failed to extract content....")

if __name__ == "__main__":
    main()
