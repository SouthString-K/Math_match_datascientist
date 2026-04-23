from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import os

class YicaiSpider:
    def __init__(self):
        edge_options = Options()
        edge_options.add_argument("--ignore-certificate-errors")
        edge_options.add_argument("--disable-blink-features=AutomationControlled")
        edge_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        
        self.driver = webdriver.Edge(options=edge_options)
        self.wait = WebDriverWait(self.driver, 10)
        self.articles = []
        self.count = 0
        self.max_count = 100
        self.json_file = "yicai_articles.json"
        
        if os.path.exists(self.json_file):
            os.remove(self.json_file)
    
    def extract_articles(self):
        """提取页面中的所有文章链接"""
        try:
            time.sleep(1)
            
            # 获取所有新闻链接
            links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/news/']")
            
            for link in links:
                if self.count >= self.max_count:
                    return
                
                try:
                    href = link.get_attribute("href")
                    title = link.text.strip()
                    
                    # 过滤有效的新闻链接
                    if href and title and href.startswith("http"):
                        # 检查是否已存在
                        if not any(article["url"] == href for article in self.articles):
                            self.articles.append({
                                "title": title,
                                "url": href
                            })
                            self.count += 1
                            print(f"[{self.count}] {title}")
                            print(f"    URL: {href}")
                            self.save_results()
                except:
                    continue
        
        except Exception as e:
            print(f"提取文章出错: {e}")
    
    def click_load_more(self):
        """点击加载更多按钮"""
        try:
            load_more_btn = self.driver.find_element(By.CLASS_NAME, "btnmore")
            self.driver.execute_script("arguments[0].scrollIntoView(true);", load_more_btn)
            time.sleep(0.5)
            load_more_btn.click()
            print("✓ 点击加载更多按钮")
            return True
        except:
            return False
    
    def run(self):
        try:
            print("正在访问一财网...")
            self.driver.get("https://www.yicai.com/news/")
            time.sleep(3)
            
            print("开始爬取文章...\n")
            
            # 先提取首页文章
            self.extract_articles()
            
            # 循环点击加载更多
            while self.count < self.max_count:
                if not self.click_load_more():
                    print("✗ 加载更多按钮不可用，爬取完成")
                    break
                
                time.sleep(2)
                self.extract_articles()
            
            print(f"\n爬取完成，共 {len(self.articles)} 条，已保存到 {self.json_file}")
        
        finally:
            self.driver.quit()
    
    def save_results(self):
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(self.articles, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    spider = YicaiSpider()
    spider.run()
