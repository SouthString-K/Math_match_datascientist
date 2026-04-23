from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import os

class FinanceSpider:
    def __init__(self):
        edge_options = Options()
        edge_options.add_argument("--ignore-certificate-errors")
        edge_options.add_argument("--ignore-ssl-errors")
        edge_options.add_argument("--disable-blink-features=AutomationControlled")
        edge_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        edge_options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Edge(options=edge_options)
        self.wait = WebDriverWait(self.driver, 10)
        self.articles = []
        self.count = 0
        self.max_count = 100
        self.json_file = "finance_articles_urls.json"
        
        if os.path.exists(self.json_file):
            os.remove(self.json_file)
    
    def get_articles_from_current_page(self):
        try:
            # 等待列表加载完成
            self.wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#newsListContent li a")))
            time.sleep(1)
            
            # 获取所有文章链接
            links = self.driver.find_elements(By.CSS_SELECTOR, "#newsListContent li a")
            
            if not links:
                print("未找到文章列表")
                return False
            
            print(f"当前页面找到 {len(links)} 条文章")
            
            # 提取所有链接信息
            articles_data = []
            for link in links:
                try:
                    title = link.text.strip()
                    href = link.get_attribute("href")
                    
                    if title and href:
                        articles_data.append({"title": title, "url": href})
                except:
                    continue
            
            # 保存提取的文章
            for article in articles_data:
                if self.count >= self.max_count:
                    return False
                
                self.articles.append(article)
                self.count += 1
                print(f"[{self.count}] {article['title']}")
                print(f"    URL: {article['url']}")
                self.save_results()
            
            return True
        
        except Exception as e:
            print(f"获取页面文章出错: {e}")
            return False
    
    def go_to_next_page(self):
        try:
            time.sleep(1)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.5)
            
            # 等待下一页按钮可点击
            next_link = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "a.next")))
            class_attr = next_link.get_attribute("class")
            
            if "unclick" in class_attr:
                print("已到最后一页")
                return False
            
            href = next_link.get_attribute("href")
            if not href:
                return False
            
            if not href.startswith("http"):
                href = "https://finance.eastmoney.com/a/" + href
            
            print(f"跳转到下一页: {href}")
            self.driver.get(href)
            time.sleep(2)
            
            return True
        
        except Exception as e:
            print(f"跳转下一页出错: {e}")
            return False
    
    def run(self):
        try:
            self.driver.get("https://finance.eastmoney.com/a/ccjdd.html")
            time.sleep(3)
            
            page = 1
            while self.count < self.max_count:
                print(f"\n========== 第 {page} 页 ==========\n")
                
                if not self.get_articles_from_current_page():
                    break
                
                if self.count >= self.max_count:
                    break
                
                if not self.go_to_next_page():
                    break
                
                page += 1
            
            print(f"\n爬取完成，共 {len(self.articles)} 条，已保存到 {self.json_file}")
        
        finally:
            self.driver.quit()
    
    def save_results(self):
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(self.articles, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    spider = FinanceSpider()
    spider.run()
