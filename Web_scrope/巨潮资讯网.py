from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import os

class AnnouncementSpider:
    def __init__(self):
        edge_options = Options()
        edge_options.add_argument("--ignore-certificate-errors")
        edge_options.add_argument("--ignore-ssl-errors")
        edge_options.add_argument("--disable-blink-features=AutomationControlled")
        edge_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        edge_options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Edge(options=edge_options)
        self.wait = WebDriverWait(self.driver, 10)
        self.actions = ActionChains(self.driver)
        self.announcements = []
        self.count = 0
        self.max_count = 100
        self.list_page_url = "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search&checkedCategory=category_gddh_szsh"
        self.json_file = "jufeng.json"
        if os.path.exists(self.json_file):
            os.remove(self.json_file)
    
    def get_announcement_urls_from_current_page(self):
        time.sleep(3)
        main_window = self.driver.current_window_handle
        
        rows = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        if not rows:
            print("未找到公告列表")
            return False
        
        print(f"当前页面找到 {len(rows)} 条公告")
        
        for idx in range(len(rows)):
            if self.count >= self.max_count:
                return False
            
            try:
                rows = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
                if idx >= len(rows):
                    break
                
                row = rows[idx]
                tds = row.find_elements(By.TAG_NAME, "td")
                
                if len(tds) < 3:
                    continue
                
                title_td = tds[2]
                title = title_td.text.strip()
                
                if not title:
                    continue
                
                windows_before = self.driver.window_handles
                current_url_before = self.driver.current_url
                
                self.actions.move_to_element(title_td).click().perform()
                time.sleep(2)
                
                windows_after = self.driver.window_handles
                current_url_after = self.driver.current_url
                
                current_url = None
                if len(windows_after) > len(windows_before):
                    new_window = [w for w in windows_after if w not in windows_before][0]
                    self.driver.switch_to.window(new_window)
                    time.sleep(1)
                    current_url = self.driver.current_urlself.driver.close()
                    self.driver.switch_to.window(main_window)
                    time.sleep(1)
                
                elif current_url_after != current_url_before and current_url_after != self.list_page_url:
                    current_url = current_url_after
                    self.driver.back()
                    time.sleep(2)
                
                else:
                    a_tag = title_td.find_element(By.TAG_NAME, "a")
                    href = a_tag.get_attribute("href")
                    if href:
                        current_url = href if href.startswith("http") else "https://www.cninfo.com.cn" + href
                
                if current_url and current_url != self.list_page_url:
                    self.announcements.append({
                        "title": title,
                        "url": current_url
                    })
                    
                    self.count += 1
                    print(f"[{self.count}] {title}")
                    print(f"    URL: {current_url}\n")
                    
                    self.save_results()
                else:
                    print(f"[跳过] {title} - 无法获取有效URL\n")
            
            except Exception as e:
                print(f"处理第 {idx} 行出错: {e}")
                try:
                    self.driver.switch_to.window(main_window)
                except:
                    pass
                continue
        
        return True
    
    def go_to_next_page(self):
        try:
            time.sleep(2)
            
            # 滚动到页面底部
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            
            # 找下一页按钮 - Element UI的btn-next
            next_button = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "button.btn-next"))
            )
            
            # 检查按钮是否被禁用
            is_disabled = next_button.get_attribute("disabled")
            if is_disabled:
                print("已到最后一页")
                return False
            
            print("找到下一页按钮，准备点击")
            
            # 滚动到按钮位置
            self.driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
            time.sleep(1)
            
            # 点击按钮
            self.driver.execute_script("arguments[0].click();", next_button)
            print("已点击下一页按钮")
            
            time.sleep(4)
            return True
        
        except Exception as e:
            print(f"跳转下一页出错: {e}")
            return False
    
    def run(self):
        try:
            self.driver.get(self.list_page_url)
            time.sleep(5)
            
            page =1
            while self.count < self.max_count:
                print(f"\n========== 第 {page} 页 ==========\n")
                
                if not self.get_announcement_urls_from_current_page():
                    break
                
                if self.count >= self.max_count:
                    break
                
                if not self.go_to_next_page():
                    break
                
                page += 1
            
            print(f"\n爬取完成，共 {len(self.announcements)} 条，已保存到 {self.json_file}")
        
        finally:
            self.driver.quit()
    
    def save_results(self):
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(self.announcements, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    spider = AnnouncementSpider()
    spider.run()
