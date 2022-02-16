from selenium import webdriver
import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia('en')

chromedriver_path = "C:/Users/harlan/Downloads/chromedriver_win32/chromedriver.exe"
num_docs = 20


browser = webdriver.Chrome(chromedriver_path)

count = 0

while count < num_docs:
    browser.get("https://en.wikipedia.org/wiki/Special:Random")
    fp = open("data_" + str(count + 1) + ".txt", "w", encoding = 'utf-8')
    if wiki_wiki.page(browser.current_url.split("/wiki/")[1]).text.replace("== References ==", "") == "":
        continue
    fp.write(wiki_wiki.page(browser.current_url.split("/wiki/")[1]).text.replace("== References ==", ""))
    count += 1
    fp.close()

browser.close()