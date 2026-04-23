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

        # 防屏蔽、防超时关键配置
        edge_options.add_argument("--no-sandbox")
        edge_options.add_argument("--disable-gpu")
        edge_options.add_argument("--ignore-certificate-errors")
        edge_options.add_argument("--ignore-ssl-errors")
        edge_options.add_argument("--disable-blink-features=AutomationControlled")
        edge_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        edge_options.add_experimental_option("useAutomationExtension", False)

        self.driver = webdriver.Edge(options=edge_options)
        self.driver.set_page_load_timeout(30)
        self.driver.implicitly_wait(10)

        self.wait = WebDriverWait(self.driver, 15)
        self.articles = []
        self.count = 0
        self.max_count = 30  # 改成30条

        # 保存路径
        self.save_folder = "D:\Math_match\codes\Web_scrope"
        self.json_file = os.path.join(self.save_folder, "东方财富30.json")

        os.makedirs(self.save_folder, exist_ok=True)

        if os.path.exists(self.json_file):
            os.remove(self.json_file)

    def get_articles_from_current_page(self):
        try:
            self.wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#newsListContent li a"))
            )
            time.sleep(1)

            links = self.driver.find_elements(By.CSS_SELECTOR, "#newsListContent li a")
            if not links:
                print("未找到文章")
                return False

            for link in links:
                if self.count >= self.max_count:
                    return False

                try:
                    title = link.text.strip()
                    href = link.get_attribute("href")

                    if title and href:
                        article = {
                            "title": title,
                            "url": href
                        }
                        self.articles.append(article)
                        self.count += 1
                        print(f"[{self.count}] {title}")
                        self.save_results()
                except:
                    continue

            return True

        except Exception as e:
            print(f"获取失败: {e}")
            return False

    def go_to_next_page(self):
        try:
            time.sleep(1)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

            next_btn = self.driver.find_element(By.CSS_SELECTOR, "a.next")
            if "unclick" in next_btn.get_attribute("class"):
                print("已到最后一页")
                return False

            href = next_btn.get_attribute("href")
            if not href.startswith("http"):
                href = "https://finance.eastmoney.com/a/" + href

            print("翻页中...")
            self.driver.get(href)
            time.sleep(2)
            return True

        except:
            print("无法翻页")
            return False

    def run(self):
        try:
            print("正在打开东方财富...")
            self.driver.get("https://finance.eastmoney.com/a/ccjdd.html")
            time.sleep(5)

            page = 1
            while self.count < self.max_count:
                print(f"\n====== 第 {page} 页 ======")
                self.get_articles_from_current_page()

                if self.count >= self.max_count:
                    break
                if not self.go_to_next_page():
                    break
                page += 1

            print("\n✅ 爬取完成！")
            print(f"共保存 {self.count} 条数据")
            print(f"文件保存在：{self.json_file}")

        except Exception as e:
            print("访问失败，请检查网络！", e)

        finally:
            self.driver.quit()

    def save_results(self):
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(self.articles, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    spider = FinanceSpider()
    spider.run()