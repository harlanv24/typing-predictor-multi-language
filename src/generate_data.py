from selenium import webdriver
import wikipediaapi
import os

wiki_wiki = wikipediaapi.Wikipedia('en')

chromedriver_path = "C:/Users/harlan/Downloads/chromedriver_win32/chromedriver.exe"
lang_codes = ['hi', 'bn']
article_counts = {'hi' : 0, 'bn' : 0}
num_docs =  500


browser = webdriver.Chrome(chromedriver_path)

# count = 0

# while count < num_docs:
#     browser.get("https://en.wikipedia.org/wiki/Special:Random")
#     fp = open("data_" + str(count + 1) + ".txt", "w", encoding = 'utf-8')
#     if wiki_wiki.page(browser.current_url.split("/wiki/")[1]).text.replace("== References ==", "") == "":
#         continue
#     fp.write(wiki_wiki.page(browser.current_url.split("/wiki/")[1]).text.replace("== References ==", ""))
#     count += 1
#     fp.close()

# browser.close()

while sum(article_counts.values()) < num_docs * 10:
    browser.get("https://en.wikipedia.org/wiki/Special:Random")
    page = wiki_wiki.page(browser.current_url.split("/wiki/")[1])    
    if page.text.replace("== References ==", "") == "":
        continue

    for lang in lang_codes:
        if article_counts[lang] == num_docs:
            continue
        if lang not in page.langlinks.keys():
            continue
        if not os.path.exists('./data/'+ lang):
            os.makedirs('./data/' + lang)
        page = page.langlinks[lang]
        fp = open('./data/' + lang + '/' + str(article_counts[lang] + 1) + "_k" + ".txt", "w", encoding = 'utf-8')
        fp.write(page.text.replace("== References ==", ""))
        article_counts[lang] += 1
        fp.close()

browser.close()