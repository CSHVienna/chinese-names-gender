import hashlib
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import itertools
import Levenshtein
from libs.cache import DiskCache
from urllib.parse import urlparse, parse_qs

NONE = ['', None]

# SEARCH_CITATIONS_URL = "https://oversea.cnki.net/kcms/detail/search.aspx?sfield=cite&code=<CODE>&dbcode=<DBCODE>&sKey=<SKEY>" # ENGLISH
SEARCH_CITATIONS_URL = "https://chn.oversea.cnki.net/kcms/detail/search.aspx?sfield=cite&code=<CODE>&dbcode=<DBCODE>&sKey=<SKEY>" # CHINESE
SEARCH_REFERENCES_URL = "https://chn.oversea.cnki.net/KCMS/detail/detail.aspx?dbcode=<DBCODE>&dbname=<DBNAME>&filename=<CODE>&uniplatform=OVERSEA&v=7jVNZ-kapPX_pS5sanKaNUIJ-aB3D-xvhlMkXrHszWncXE1phNdikrZ8rwbTNuJJ"
NPAPERS_PER_BOX = 10
ENGLISH = {'题名':'Title', '作者':'Author', '来源':'Source', '发表时间':'Publication Date', '数据库':'Database', '被引':'Cites',
           '下载':'Downloads', '操作':'Options', '编号':'Result Rank',
           '期刊':'Journal', '博士':'Thesis', '硕士':'Dissertation', '图书':'Books',
           '中国会议':'ChineseConf', '中外文题录':'OtherLang', '国际会议':'InternationalConf', '国际期刊':'InternationalJournal'}
CHINESE = {'Title':'篇名', 'Title2':'题名', 'Journal':'期刊',
           'Cites':'被引', 'Options':'操作', 'Find':'共', 'NEXT':'下一页', 'OtherLang':'中外文题录'}

class CNKI(DiskCache):

    def __init__(self, cache_dir):
        super().__init__(cache_dir=cache_dir)
        self.chrome_options = None
        self.service = None
        # self.search_url = "https://oversea.cnki.net/index" # ENGLISH
        self.search_url = "https://chn.oversea.cnki.net/index/"  # CHINESE
        self.search_query = None

    def setup(self, chromedriver_path='/usr/local/bin/chromedriver'):
        # Initialize the webdriver
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")  # Enable headless mode
        self.chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration
        self.chrome_options.add_argument("--window-size=1920x1080")  # Set window size to avoid issues with some websites
        self.chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
        self.chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems in Docker

        # Specify the path to chromedriver executable if not in PATH
        self.service = Service(chromedriver_path)  # Update with your ChromeDriver path

    def get_references(self, obj_paper):
        self.search_query = obj_paper['Title']
        self.reference_count = 0
        self.search_url = SEARCH_REFERENCES_URL.replace("<CODE>",obj_paper['scode']).replace("<DBCODE>",obj_paper['sdb']).replace('<DBNAME>',obj_paper['dbname'])
        id_request = CNKI.generate_cache_key(self.search_url)
        result = self.get_or_generate(id_request, self._do_reference_search)
        return result

    def _do_reference_search(self):
        # Initialize the WebDriver with the specified options
        driver = webdriver.Chrome(service=self.service, options=self.chrome_options)
        # driver.set_page_load_timeout(max(60, self.citations_count*5))
        # driver.set_script_timeout(max(60, self.citations_count*5))
        driver.set_page_load_timeout(60)
        driver.set_script_timeout(60)

        # Step 1: Open the website
        driver.get(self.search_url)

        seq_id = 1
        kinds = set()
        results = []

        ######################################
        # Main crawler
        def extract_references(kind_pagination=None):
            nonlocal seq_id
            try:
                iframe = driver.find_element(By.ID, "frame1")
                driver.switch_to.frame(iframe)
            except Exception as ex:
                # print(ex)
                pass
            # div_result = driver.find_element(By.ID, "frame1")

            essay_boxes = driver.find_elements(By.CLASS_NAME, "essayBox")

            for box in essay_boxes:
                kind, npapers = box.find_element(By.TAG_NAME, "div").text.split(CHINESE["Find"])
                kind = kind.strip()
                npapers = int(npapers.replace('条','').strip())

                if kind_pagination is None:
                    self.reference_count += npapers # actual counts

                if npapers > NPAPERS_PER_BOX and kind_pagination is None:
                    kinds.add(kind)

                if kind_pagination is None or kind_pagination==kind:
                    papers = box.find_elements(By.TAG_NAME, "li")

                    for paper in papers:
                        tmp = paper.text.split('\n')[-1]
                        filename, dbcode, dbname = None, None, None

                        if len(tmp.split('.')) == 2:
                            title = CNKI.cleanupstr(tmp.split('.')[0])
                            year = CNKI.cleanupstr(tmp.split(',')[-1])
                            authors = CNKI.cleanupstr(tmp.split(',')[-2])
                            journal = CNKI.cleanupstr(tmp.split(',')[-3])

                        elif kind != CHINESE['OtherLang']:
                            if len(tmp.split('.')) > 4:
                                title = CNKI.cleanupstr(tmp.split('.')[0])
                                year = CNKI.cleanupstr(tmp.split('.')[-1])
                                journal = CNKI.cleanupstr(tmp.split('.')[-2])
                                authors = CNKI.cleanupstr(tmp.replace(title,'').replace(year,'').replace(journal,''))
                            else:
                                title = CNKI.cleanupstr(tmp.split('.')[0])
                                authors = CNKI.cleanupstr(tmp.split('.')[1])
                                journal = CNKI.cleanupstr(tmp.split('.')[2]) if kind==CHINESE['Journal'] else CNKI.cleanupstr(tmp.split('.')[2][:-5])
                                year = CNKI.cleanupstr(tmp.split('.')[3]) if kind==CHINESE['Journal'] else CNKI.cleanupstr(tmp.split('.')[2][-4:])

                            # Parse the URL
                            try:
                                url = paper.find_element(By.TAG_NAME, 'a').get_attribute('href')
                                parsed_url = urlparse(url)
                                query_params = parse_qs(parsed_url.query)
                                filename = query_params.get('filename', [None])[0]
                                dbcode = query_params.get('dbcode', [None])[0]
                                dbname = query_params.get('dbname', [None])[0]
                            except Exception as ex:
                                # print(ex)
                                pass

                        else:
                            title = None
                            authors = CNKI.cleanupstr(tmp.split('.')[0])
                            year = CNKI.cleanupstr(tmp.split('.')[-1])
                            journal = CNKI.cleanupstr(tmp.replace(authors,'').replace(year,''))

                        # main: https://oversea.cnki.net/kns/Detail?sfield=fn&QueryID=0&CurRec=1&recid=&FileName=WLZZ407.001&DbName=CJFD9495&DbCode=CJFD&yx=&pr=&URLID=
                        # main cites: https://oversea.cnki.net/kns/Detail?sdb=CJFD&sfield=cite&skey=%e4%ba%9a%e7%9a%ae%e7%a7%92%e5%85%89%e6%95%b4%e6%b5%81%e6%95%88%e5%ba%94&scode=WLZZ407.001
                        # cite: https://oversea.cnki.net/kcms/detail/detail.aspx?filename=GXXB202215002&dbcode=CJFQ&dbname=DKFXTEMP&v=

                        results.append({
                            "order": seq_id,
                            "title": title,
                            "authors": authors,
                            "journal": journal,
                            "year": year,
                            "kind": ENGLISH[kind],
                            "filename": filename,
                            "dbcode": dbcode,
                            "dbname": dbname,
                        })
                        seq_id += 1
                time.sleep(1)
        ######################################

        # Click on references (to reload)
        try:
            time.sleep(30)
            ref_button = driver.find_element(By.ID, "rl1")
            ref_button.click()
        except Exception as ex:
            print('ERROR 1:', ex)
            return None

        try:
            self.reference_count = int(driver.find_element(By.ID, 'rc1').text.strip(')').strip('('))
        except Exception as ex:
            print('ERROR 2:', ex)
            self.reference_count = 0

        if self.reference_count > 0:
            # waiting times based on # of papers to load
            wait = max(30, int(self.reference_count / 10))
            minwait = 5

            # get citations current page
            try:
                time.sleep(wait)
                self.reference_count = 0 # the reference count from "rc1" sometimes does not show what it actually has.
                extract_references()
                flag = True
            except Exception as ex:
                print(ex)
                flag = False

            # get citations with pagination
            if flag:
                for kind in kinds:
                    while True:
                        try:
                            xpath = f"//div[@class='essayBox'][.//div[@class='dbTitle' and text()='{kind}']]//div[@class='pageBar']//a[text()='{CHINESE['NEXT']}']"
                            next_button = driver.find_element(By.XPATH, xpath)
                            next_button.click()
                            time.sleep(minwait)
                            # WebDriverWait(driver, wait).until(EC.staleness_of(next_button))
                            WebDriverWait(driver, wait).until(EC.visibility_of_all_elements_located((By.CLASS_NAME, "essayBox")))
                            time.sleep(minwait)
                            extract_references(kind_pagination=kind)
                        except Exception as ex:
                            #print(ex)
                            break

            driver.quit()

        print(f"reference results ({self.search_query}): {len(results)} (out of {self.reference_count})")
        return None if len(results) < self.reference_count else results

    def get_citations(self, obj_paper):
        self.search_query = obj_paper['Title']
        self.citations_count = int(obj_paper['Cites']) if obj_paper['Cites'] not in NONE else 0
        result = None
        if self.citations_count > 0:
            self.search_url = SEARCH_CITATIONS_URL.replace("<CODE>",obj_paper['scode']).replace("<DBCODE>",obj_paper['sdb']).replace("<SKEY>",obj_paper['Title'])
            id_request = CNKI.generate_cache_key(self.search_url)
            result = self.get_or_generate(id_request, self._do_citations_search)
        return result

    def _do_citations_search(self):
        # Initialize the WebDriver with the specified options
        driver = webdriver.Chrome(service=self.service, options=self.chrome_options)
        driver.set_page_load_timeout(max(60, self.citations_count*5))
        driver.set_script_timeout(max(60, self.citations_count*5))

        # Step 1: Open the website
        driver.get(self.search_url)

        seq_id = 1
        kinds = set()
        results = []

        ######################################
        # Main crawler
        def extract_citations(kind_pagination=None):
            nonlocal seq_id
            div_result = driver.find_element(By.ID, "divResult")
            essay_boxes = div_result.find_elements(By.CLASS_NAME, "essayBox")

            for box in essay_boxes:
                kind, npapers = box.find_element(By.TAG_NAME, "div").text.split(CHINESE["Find"])
                kind = kind.strip()
                npapers = int(npapers.replace('条','').strip())

                if npapers > NPAPERS_PER_BOX and kind_pagination is None:
                    kinds.add(kind)

                if kind_pagination is None or kind_pagination==kind:
                    papers = box.find_elements(By.TAG_NAME, "li")

                    for paper in papers:
                        tmp = paper.text.split('\n')[-1]
                        if kind != CHINESE['OtherLang']:
                            title = CNKI.cleanupstr(tmp.split('.')[0])
                            authors = CNKI.cleanupstr(tmp.split('.')[1])
                            journal = CNKI.cleanupstr(tmp.split('.')[2]) if kind==CHINESE['Journal'] else CNKI.cleanupstr(tmp.split('.')[2][:-5])
                            year = CNKI.cleanupstr(tmp.split('.')[3]) if kind==CHINESE['Journal'] else CNKI.cleanupstr(tmp.split('.')[2][-4:])

                            # Parse the URL
                            try:
                                url = paper.find_element(By.TAG_NAME, 'a').get_attribute('href')
                                parsed_url = urlparse(url)
                                query_params = parse_qs(parsed_url.query)
                                filename = query_params.get('filename', [None])[0]
                                dbcode = query_params.get('dbcode', [None])[0]
                                dbname = query_params.get('dbname', [None])[0]
                            except:
                                filename, dbcode, dbname = None, None, None
                        else:
                            title, filename, dbcode, dbname = None, None, None, None
                            if len(tmp.split('.')) > 3:
                                authors = CNKI.cleanupstr(tmp.split('.')[-3])
                                year = CNKI.cleanupstr(tmp.split('.')[-1])
                                journal = CNKI.cleanupstr(tmp.split('.')[-2])
                            else:
                                authors = CNKI.cleanupstr(tmp.split('.')[0])
                                year = CNKI.cleanupstr(tmp.split('.')[-1])
                                journal = CNKI.cleanupstr(tmp.replace(authors,'').replace(year,''))

                        # main: https://oversea.cnki.net/kns/Detail?sfield=fn&QueryID=0&CurRec=1&recid=&FileName=WLZZ407.001&DbName=CJFD9495&DbCode=CJFD&yx=&pr=&URLID=
                        # main cites: https://oversea.cnki.net/kns/Detail?sdb=CJFD&sfield=cite&skey=%e4%ba%9a%e7%9a%ae%e7%a7%92%e5%85%89%e6%95%b4%e6%b5%81%e6%95%88%e5%ba%94&scode=WLZZ407.001
                        # cite: https://oversea.cnki.net/kcms/detail/detail.aspx?filename=GXXB202215002&dbcode=CJFQ&dbname=DKFXTEMP&v=

                        results.append({
                            "order": seq_id,
                            "title": title,
                            "authors": authors,
                            "journal": journal,
                            "year": year,
                            "kind": ENGLISH[kind],
                            "filename": filename,
                            "dbcode": dbcode,
                            "dbname": dbname,
                        })
                        seq_id += 1
                time.sleep(1)
        ######################################


        wait = max(5, int(self.citations_count / 10))
        minwait = 3

        # get citations current page
        try:
            time.sleep(wait)
            extract_citations()
            flag = True
        except Exception as ex:
            print(ex)
            flag = False

        # get citations with pagination
        if flag:
            for kind in kinds:
                while True:
                    try:
                        xpath = f"//div[@id='divResult']//div[@class='essayBox'][.//div[@class='dbTitle' and text()='{kind}']]//div[@class='pageBar']//a[text()='{CHINESE['NEXT']}']"
                        next_button = driver.find_element(By.XPATH, xpath)
                        next_button.click()
                        time.sleep(minwait)
                        # WebDriverWait(driver, wait).until(EC.staleness_of(next_button))
                        WebDriverWait(driver, wait).until(EC.visibility_of_all_elements_located((By.CLASS_NAME, "essayBox")))
                        time.sleep(minwait)
                        extract_citations(kind_pagination=kind)
                    except Exception as ex:
                        #print(ex)
                        break

        driver.quit()

        print(f"citation results ({self.search_query}): {len(results)} (out of {self.citations_count})")
        return None if len(results) < self.citations_count else results


    def search_paper_by_title(self, title):
        id_request = CNKI.generate_cache_key(self.search_url, title)
        self.search_query = title
        result = self.get_or_generate(id_request, self._do_paper_search)
        return result

    def _do_paper_search(self):
        data = {}

        try:
            # Initialize the WebDriver with the specified options
            driver = webdriver.Chrome(service=self.service, options=self.chrome_options)

            # Step 1: Open the website
            driver.get(self.search_url)

            # Step 2: Change the span text from "Subject" to "Title" using JavaScript
            script = f"""document.querySelector("#DBFieldBox .sort-default span").textContent = "{CHINESE['Title']}";"""
            driver.execute_script(script)

            # Step 3: Type in the input
            search_input = driver.find_element(By.CLASS_NAME, "search-input")
            search_input.send_keys(self.search_query)

            # Step 4: Click on the search button
            search_button = driver.find_element(By.ID, "search")
            search_button.click()

            # Step 5: Wait for the results to load
            time.sleep(5)  # Wait for 5 seconds to ensure results are loaded

            # Step 6: Parse results

            # Extract table headers
            results_table = driver.find_element(By.CLASS_NAME, "result-table-list")
            headers = results_table.find_element(By.TAG_NAME, "thead").find_elements(By.TAG_NAME, "th")
            header_names = [header.text for header in headers]

            # Extract table rows
            body = results_table.find_element(By.TAG_NAME, "tbody")
            rows = body.find_elements(By.TAG_NAME, "tr")

            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                for header, cell in zip(*(header_names, cells)):
                    if header in ['Options', CHINESE['Options']]:
                        continue
                    header = '编号' if header in NONE else header
                    data[ENGLISH[header]] = cell.text

                    # DBNAME for references
                    if header in ['Title', CHINESE['Title'], CHINESE['Title2']]:
                        # get dbanme from URL in Tile
                        data['dbname'] = None
                        try:
                            link = cell.find_element(By.TAG_NAME, "a")
                            url_details = link.get_attribute("href") if link else None
                            if url_details not in NONE:
                                # Parse the URL
                                parsed_url = urlparse(url_details)

                                # Parse the query parameters
                                query_params = parse_qs(parsed_url.query)

                                # Extract the values for scode and sdb
                                dbname = query_params.get('DbName', [None])[0]
                                data['dbname'] = dbname

                        except Exception as ex:
                            # print(ex)
                            pass

                    # SCODE and SDB for Citations
                    if header in ['Cites', CHINESE['Cites']]:
                        # Check if the cell contains an <a> tag
                        # Get scode, sdb from URL in Cites

                        # data['url_citations'] = None
                        data['Cites'] = 0 if data['Cites'] in NONE else data['Cites']
                        data['scode'] = None
                        data['sdb'] = None
                        try:
                            if data[ENGLISH[header]] not in NONE:
                                link = cell.find_element(By.TAG_NAME, "a")
                                url_citations = link.get_attribute("href") if link else None
                                # data['url_citations'] = url_citations

                                if url_citations not in NONE:

                                    # Parse the URL
                                    parsed_url = urlparse(url_citations)

                                    # Parse the query parameters
                                    query_params = parse_qs(parsed_url.query)

                                    # Extract the values for scode and sdb
                                    scode = query_params.get('scode', [None])[0]
                                    data['scode'] = scode

                                    sdb = query_params.get('sdb', [None])[0]
                                    data['sdb'] = sdb

                        except Exception as ex:
                            # print(ex)
                            pass

                if Levenshtein.distance(data['Title'].lower(), self.search_query.lower()) < 5 and \
                        Levenshtein.ratio(data['Title'].lower(), self.search_query.lower()) > 0.9:
                    break

        finally:
            # Close the driver
            driver.quit()

        return data

    @staticmethod
    def generate_cache_key(url, query=None):
        if query is None:
            return hashlib.md5(f"{url}".encode('utf-8')).hexdigest()
        return hashlib.md5(f"{url}/title={query}".encode('utf-8')).hexdigest()

    @staticmethod
    def get_user_agents(fn_user_agents="../../data/user_agents.txt"):
        user_agents = ['Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36']
        with open(fn_user_agents, 'r') as f:
            user_agents = [line.strip().replace('\n', '') for line in f.readlines()]
        user_agents = itertools.cycle(user_agents)
        return user_agents

    @staticmethod
    def cleanupstr(s):
        return s.replace('[J]', '').replace('[D]', '').replace('[T]', '').replace('[M]', '').replace('[A]', '').replace('[C]', '').strip('.').strip().strip('.').strip()
