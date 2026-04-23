from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
import time
import json

class PolicySpider:
    def __init__(self):
        edge_options = Options()
        edge_options.add_argument("--ignore-certificate-errors")
        edge_options.add_argument("--ignore-ssl-errors")
        edge_options.add_argument("--disable-blink-features=AutomationControlled")
        edge_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        edge_options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Edge(options=edge_options)
        self.policies = []
        self.count = 0
        self.max_count = 100
    
    def get_policy_urls_from_current_page(self):
        """从当前页面获取所有政策链接"""
        try:
            # 等待页面加载
            time.sleep(3)
            
            # 尝试多个选择器
            selectors = [
                "div.news-list2 li a",
                "ul.news-list li a",
                "div.list-item a",
                "li a[href*='zhengce']"
            ]
            
            policy_elements = []
            for selector in selectors:
                try:
                    policy_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if policy_elements:
                        print(f"找到选择器: {selector}")
                        break
                except:
                    continue
            
            if not policy_elements:
                print("未找到政策列表元素")
                return False
            
            print(f"当前页面找到 {len(policy_elements)} 个政策")
            
            # 提前收集所有链接信息，避免 stale element 问题
            links_info = []
            for element in policy_elements:
                try:
                    title = element.text.strip()
                    href = element.get_attribute("href")
                    if title and href:
                        links_info.append({"title": title, "href": href})
                except:
                    continue
            
            # 逐个处理链接
            for link_info in links_info:
                if self.count >= self.max_count:
                    return False
                
                try:
                    title = link_info["title"]
                    href = link_info["href"]
                    
                    # 在新标签页打开
                    self.driver.execute_script("window.open(arguments[0]);", href)
                    
                    # 切换到新标签页
                    self.driver.switch_to.window(self.driver.window_handles[-1])
                    time.sleep(3)  # 等待页面加载
                    
                    # 获取当前页面的URL
                    current_url = self.driver.current_url
                    
                    self.policies.append({
                        "title": title,
                        "url": current_url,
                        "original_href": href
                    })
                    
                    self.count += 1
                    print(f"[{self.count}] {title}")
                    print(f"    URL: {current_url}\n")
                    
                    # 关闭当前标签页，回到主页面
                    self.driver.close()
                    self.driver.switch_to.window(self.driver.window_handles[0])
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"处理政策出错: {e}")
                    try:
                        self.driver.switch_to.window(self.driver.window_handles[0])
                    except:
                        pass
                    continue
            
            return True
        
        except Exception as e:
            print(f"获取政策列表出错: {e}")
            return False
    
    def go_to_next_page(self):
        """点击下一页按钮"""
        try:
            # 尝试多个下一页选择器
            selectors = [
                "a:contains('下一页')",
                "//a[contains(text(), '下一页')]",
                "div.pager a:last-child",
                "div.page a[href*='page']",
                "a.page-next"
            ]
            
            next_button = None
            
            # 先尝试 CSS 选择器
            css_selectors = [
                "a:contains('下一页')",
                "div.pager a:last-child",
                "div.page a[href*='page']",
                "a.page-next"
            ]
            
            for selector in css_selectors:
                try:
                    next_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if next_button:
                        print(f"找到下一页按钮 (CSS): {selector}")
                        break
                except:
                    continue
            
            # 如果 CSS 没找到，尝试 XPath
            if not next_button:
                xpath_selectors = [
                    "//a[contains(text(), '下一页')]",
                    "//a[text()='下一页']",
                    "//div[@class='pager']//a[last()]"
                ]
                
                for xpath in xpath_selectors:
                    try:
                        next_button = self.driver.find_element(By.XPATH, xpath)
                        if next_button:
                            print(f"找到下一页按钮 (XPath): {xpath}")
                            break
                    except:
                        continue
            
            if not next_button:
                print("未找到下一页按钮")
                # 打印页面中所有的链接，用于调试
                all_links = self.driver.find_elements(By.TAG_NAME, "a")
                print(f"页面中共有 {len(all_links)} 个链接")
                for i, link in enumerate(all_links[-5:]):
                    print(f"  链接 {i}: {link.text} - {link.get_attribute('href')}")
                return False
            
            # 检查是否禁用
            class_attr = next_button.get_attribute("class")
            if class_attr and "disabled" in class_attr:
                print("已到最后一页")
                return False
            
            self.driver.execute_script("arguments[0].click();", next_button)
            time.sleep(3)  # 等待页面加载
            return True
        
        except Exception as e:
            print(f"跳转下一页出错: {e}")
            return False
    
    def run(self):
        """主爬虫流程"""
        try:
            self.driver.get("https://www.gov.cn/zhengce/zuixin/")
            time.sleep(5)  # 等待页面加载
            
            page = 1
            while self.count < self.max_count:
                print(f"\n========== 第 {page} 页 ==========\n")
                
                # 爬取当前页面的政策
                if not self.get_policy_urls_from_current_page():
                    break
                
                # 如果已达到目标数量，停止
                if self.count >= self.max_count:
                    break
                
                # 跳转到下一页
                if not self.go_to_next_page():
                    break
                
                page += 1
            
            # 保存结果
            self.save_results()
        
        finally:
            self.driver.quit()
    
    def save_results(self):
        """保存爬取结果到JSON文件"""
        with open("policies.json", "w", encoding="utf-8") as f:
            json.dump(self.policies, f, ensure_ascii=False, indent=2)
        
        print(f"\n\n========== 爬取完成 ==========")
        print(f"共爬取 {len(self.policies)} 条政策")
        print(f"结果已保存到 policies.json")

if __name__ == "__main__":
    spider = PolicySpider()
    spider.run()
