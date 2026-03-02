
import os
import random
import requests
import pandas as pd
import dask.dataframe as dd
from bs4 import BeautifulSoup
from dask.diagnostics import ProgressBar

from libs import ios

URL_CHINESE_TOOL = 'https://www.chinese-tools.com/tools/gender-guesser.html'
HTTP_OUTPUT = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
CHUNK_SIZE = 500
SEPARATOR = '#####'

class ChineseTools(object):

  # https://www.chinese-tools.com/tools/gender-guesser.html
  # POST
  # name='弘'

  # <div class="ctgender_result"><big>弘</big> (Hóng) is a male name</div>
  # <div class="ctgender_annotation">
  #   <div class="ctanno_result">
  #     <div class="ctanno_text">
  #       <span class="ctanno_tchinese">&#24344;</span><span class="ctanno_tnumber">1</span>
  #     </div>
  #     <div class="ctanno_vocabulary">
  #       <span class="ctanno_vnumber">1.</span>
  #       <span class="ctanno_vchinese">&#24344;</span>
  #       <span class="ctanno_vpinyin">h&#243;ng </span>
  #       <span class="ctanno_vdef">great / liberal</span>
  #     </div>
  #   </div>
  # </div>


  # The URL you want to scrape
  url = URL_CHINESE_TOOL
  
  def __init__(self, fn_full_names,
                  fn_given_names, 
                  fn_ch_given_names,
                  fn_en_given_names,
                  fn_ch_given_names_gender,
                  fn_en_given_names_gender, 
                  fn_full_names_gender, 
                  resource_dir=None,
                  cache_dir=None):
    self.fn_full_names = fn_full_names
    self.fn_given_names = fn_given_names
    self.fn_ch_given_names = fn_ch_given_names
    self.fn_en_given_names = fn_en_given_names
    self.fn_ch_given_names_gender = fn_ch_given_names_gender
    self.fn_en_given_names_gender = fn_en_given_names_gender
    self.fn_full_names_gender = fn_full_names_gender
    self.resource_dir = resource_dir
    self.cache_dir = cache_dir
    self.proxies = ios.read_proxies(self.resource_dir)
    self.user_agents = ios.read_proxies(self.resource_dir)
    self.accept_output = HTTP_OUTPUT
    self.df_full_names = None
    self.df_given_names = None
    self.df_ch_given_names = None
    self.df_en_given_names = None
    self.df_ch_gender = None
    self.df_en_gender = None
    self.df_full_names_gender = None
    self.__load_data()

  def __load_data(self):
    self.df_full_names = ios.load_csv(self.fn_full_names)
    self.df_given_names = ios.load_csv(self.fn_given_names)

  def load_unique_english_given_names(self):
    self.df_en_given_names = ios.load_csv(self.fn_en_given_names)
    if self.df_en_given_names is None:
      english_unique_names = self.df_given_names['english'].str.split(SEPARATOR).explode().unique()
      self.df_en_given_names = pd.DataFrame({'name':english_unique_names}).sort_values('name').dropna(subset='name').reset_index(drop=True)
      self.df_en_given_names.to_csv(self.fn_en_given_names, index=True) 

  def load_unique_chinese_given_names(self):
    self.df_ch_given_names = ios.load_csv(self.fn_ch_given_names)
    if self.df_ch_given_names is None:
      chinese_unique_names = self.df_given_names['givenname'].unique()
      self.df_ch_given_names = pd.DataFrame({'name':chinese_unique_names}).sort_values('name').dropna(subset='name').reset_index(drop=True)
      self.df_ch_given_names.to_csv(self.fn_ch_given_names, index=True)

  def batch_get_genders_from_english_name(self, verbose=True):
    
    # load gender file
    if self.fn_en_given_names_gender is not None and os.path.exists(self.fn_en_given_names_gender):
      if verbose:
        print(f'[INFO] {self.fn_en_given_names_gender} loading...')
      self.df_en_gender = pd.read_csv(self.fn_en_given_names_gender, index_col=0)
      return 
    
    # Infer gender of english names from scratch

    if verbose:
      print(self.df_full_names.head(2))
      print(self.df_ch_gender.head(2))
      print(f"[INFO] df_full: {self.df_full_names.shape}, self.df_ch_gender: {self.df_ch_gender.shape}")

    # Adding gender to full names
    self.df_full_names_gender = self.df_full_names.set_index('givenname').join(self.df_ch_gender[['givenname','gender','english']].set_index('givenname')).reset_index()
    self.df_full_names_gender = self.df_full_names_gender[['fullname','familyname','givenname','english','gender']]
    assert self.df_full_names.shape[0] == self.df_full_names_gender.shape[0]

    # Expanding English given names
    tmp = self.df_ch_gender.apply(lambda row: str(row.english).split(';'), axis=1).explode().reset_index(name='english').rename(columns={'index':'index_original'})
    df_en = self.df_ch_gender.drop(columns=['english']).join(other=tmp.set_index('index_original'), how='inner').reset_index().rename(columns={'index':'index_original'})
    
    for s in [1,2,3]:
      g = df_en.groupby('english').filter(lambda x: len(set(x.gender)) >= s).sort_values('english').shape[0]
      if verbose:
        print(f'[INFO] English names with >= {s} gener: {g}')
    
    self.df_en_gender = pd.DataFrame(columns=['english',
                                              'chinese_names', 'piyin_names', 'piyin_wacc_names',
                                              'original_indexes',
                                              'ac_prob_female', 'ac_prob_male', 'ac_gender',  # based on AC-ademics frequencies
                                              'na_prob_female', 'na_prob_male', 'na_gender',  # based on NA-me frequencies
                                              'gender'                                        # consensus
                                              ]) # based on ch-en frequencies
    precision = 3
    all_genders = ['female', 'male']
    for english_name, group_df in df_en.groupby('english'):
      # among names
      g_ch = group_df.query("gender in @all_genders")
      total = g_ch.shape[0]
      na_pf = round(g_ch.query("gender=='female'").shape[0]/total,precision) if total > 0 else None
      na_pm = round(1 - na_pf,precision) if total > 0 else None
      na_gender = 'female' if na_pf == 1 else 'male' if na_pm == 1 else 'unisex' if total > 0 else None

      # among people
      ch_given_names = group_df.givenname.unique()
      ac_df = self.df_full_names_gender.query("givenname in @ch_given_names and gender in @all_genders")
      total = ac_df.shape[0]
      ac_pf = round(ac_df.query("gender=='female'").shape[0]/total,precision) if total > 0 else None
      ac_pm = round(1 - ac_pf,precision) if total > 0 else None
      ac_gender = 'female' if ac_pf == 1 else 'male' if ac_pm == 1 else 'unisex' if total > 0 else None

      # consensus gender
      gender =  None if (na_pf is None and ac_pf is None) else na_gender if na_gender == ac_gender else 'unisex' if (na_gender=='unisex' or ac_gender=='unisex') else 'ambiguous'

      # final
      # genders = '|'.join(group_df.gender.astype(str))
      cnames = '|'.join(group_df.givenname.astype(str))
      pnames = '|'.join(group_df.pinyin.astype(str))
      pwnames = '|'.join(group_df.pinyin_wacc.astype(str))
      indexes_original = '|'.join(group_df.index_original.astype(str))

      obj = {'english':english_name,
            'chinese_names':cnames,
            'piyin_names':pnames,
            'piyin_wacc_names':pwnames,
            'original_indexes':indexes_original,
            'ac_prob_female':ac_pf,
            'ac_prob_male':ac_pm,
            'ac_gender':ac_gender,
            'na_prob_female':na_pf,
            'na_prob_male':na_pm,
            'na_gender':na_gender,
            'gender':gender,
            }

      self.df_en_gender = pd.concat([self.df_en_gender,pd.DataFrame(obj,index=[0])], ignore_index=True)

    if verbose:
      #21446
      #20861
      print(f"[INFO] English names with gender: {self.df_en_gender.shape }")

    # Validations
    dif = self.df_en_gender.query("na_gender!=ac_gender").shape[0]  # gender differs from ACademic frequencies and NAme frequencies
    equ = self.df_en_gender.query("na_gender==ac_gender").shape[0]  # gender consensus from ACademic frequencies and NAme frequencies
    critical = self.df_en_gender.query("na_gender!=ac_gender and na_gender!='unisex' and ac_gender!='unisex'").shape[0]
    assert dif == 0 and equ == self.df_en_gender.shape[0] and critical == 0
    

    # Save
    if self.fn_en_given_names_gender is not None:
      self.df_en_gender.to_csv(self.fn_en_given_names_gender)
      self.df_full_names_gender.to_csv(self.fn_full_names_gender)


  def batch_get_genders_from_chinese_name(self, 
                                          cache_dir, chunk_size=None, 
                                          n_jobs=None, verbose=True):
    
    # if it already exist, load
    if os.path.exists(self.fn_ch_given_names_gender):
      if verbose:
        print(f'[INFO] {self.fn_ch_given_names_gender} loading...')
      self.df_ch_gender = pd.read_csv(self.fn_ch_given_names_gender, index_col=0)
      return
    
    # else, query (or read cache)
    df_given_names = pd.read_csv(self.fn_given_names)

    n_jobs = os.cpu_count() if n_jobs is None else n_jobs
    if verbose:
      print(f'[INFO] {n_jobs} parallel jobs.')

    os.makedirs(cache_dir, exist_ok=True)
    if verbose:
      print(f'[INFO] results will be saved in: {cache_dir}')

    chunk_size = CHUNK_SIZE if chunk_size is None else chunk_size
    if verbose:
      print(f'[INFO] chunk size: {chunk_size}')

    # For every chunk
    chunk_id = -1
    for start in range(0, len(df_given_names), chunk_size):
      chunk_id += 1

      # if exists, skip
      fn = os.path.join(cache_dir, f'chunk_{chunk_id}.csv')
      if os.path.exists(fn):
        if verbose:
          print(f'[INFO] chunk_id {chunk_id} already done.')
        continue

      # else, process chunk
      try:
        end = start + chunk_size
        chunk = df_given_names.iloc[start:end]
        result_df = self.process_chunk(chunk, n_jobs)
        result_df.to_csv(fn)
        
      except Exception as ex:
        print(f'[ERROR] {ex} (chunk_id: {chunk_id})')
    
    # return all
    df_from_cache = ios.merge_dataframes_from_folder(cache_dir)

    # Not retrieved in the script - but later directly on the website
    df_from_cache.loc[15278, 'pinyin_name'] = 'Zūn Jiàn'
    df_from_cache.loc[15278, 'gender'] = 'male'
    df_from_cache.loc[2092, 'pinyin_name'] = 'Méng Tāo'
    df_from_cache.loc[2092, 'gender'] = 'male'
    df_from_cache.loc[37423, 'gender'] = 'male'

    if verbose:
      sc = ['萌涛','尊建','福昊']
      print(f"[INFO] {df_from_cache.query("chinese_name in @sc")}")

    # JOIN: append to gender and name, the piyin, accents, etc
    self.df_ch_gender = pd.read_csv(self.fn_given_names).join(df_from_cache)

    if verbose:
      print(f"[INFO] df_ch_gende.shape {self.df_ch_gender.shape}")

    # curate columns
    self.df_ch_gender.drop(columns=['chinese_name'], inplace=True)
    self.df_ch_gender.rename(columns={'pinyin_name':'pinyin_wacc'}, inplace=True)
    self.df_ch_gender = self.df_ch_gender[['givenname','pinyin_wacc','pinyin','english','gender']].copy()
    
    # names without gender
    if verbose:
      print(f"[INFO] {self.df_ch_gender.query("gender not in ['male','female']").shape}")
    assert self.df_ch_gender.query("gender not in ['male','female']").shape[0] == 11

    # save
    self.df_ch_gender.to_csv(self.fn_ch_given_names_gender)

  def process_chunk(self, df, n_jobs):
    # Convert the pandas DataFrame to a Dask DataFrame
    ddf = dd.from_pandas(df, npartitions=n_jobs)

    # Operations on Dask DataFrames are lazy; they're not computed until you explicitly ask
    result = ddf.apply(self.chinese_tools_gender_guesser_wrapper, axis=1, result_type='expand', meta={0: str, 1: str, 2:str})
    result.columns = ['chinese_name', 'pinyin_name', 'gender']
    
    # Compute the result with a progress bar
    with ProgressBar():
        computed_result_df = result.compute()

    return computed_result_df

  def chinese_tools_gender_guesser_wrapper(self, row):
    return self.chinese_tools_gender_guesser(name=row['givenname'])

  def get_cache_filename(self, name):
    return os.path.join(self.cache_dir, f'{name}.json')
  
  def load_from_cache(self, name):
    if self.cache_dir is None:
      return (None, None, None)

    fn = self.get_cache_filename(name)
    if os.path.exists(fn):
      obj = ios.read_json(fn)
      return (obj['chinese'], obj['piyin'], obj['gender'])
    
    return (None, None, None)
  
  def save_to_cache(self, name, ch, py, g):
    if self.cache_dir is None:
      return
    fn = self.get_cache_filename(name)
    ios.save_json({'chinese':ch, 'piyin':py, 'gender':g}, fn)
    
  def chinese_tools_gender_guesser(self, name, tries=1):
    
    if type(name) != str or str(name) == '':
      return (None, None, None)

    responses = []
    name = str(name).strip()

    # load from cache if exists
    ch, py, g = self.load_from_cache(name)
    if ch is not None:
      # print('loaded from cache')
      return (ch, py, g)
    
    # The data to be sent with the POST request
    data = {'name': name}

    while True:
      try:
        user_agent = random.choice(self.user_agents)
        proxy = random.choice(self.proxies)
        headers = {'User-Agent': user_agent, 'Accept':self.accept_output}
        proxies_dict = {'http': proxy, 'https': proxy}

        # Sending a POST request to the URL
        response = requests.post(self.url, data=data, headers=headers) #, proxies=proxies_dict)

        # Checking if the request was successful
        if response.status_code == 200:
            # Parsing the HTML content of the response
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extracting content from the div with class 'ctg_result'
            content_div = soup.find('div', class_='ctgender_result')

            if content_div is not None:
              content_text = content_div.get_text(strip=True)  # strip=True removes leading/trailing whitespace
              responses.append(content_text)

              # <chinese> (<piyin>) is a <gender> name
              tmp = content_text.split(' is a ')
              
              tmp_name = tmp[0].split('(')
              if len(tmp_name) == 2:
                ch, py = tmp_name[0], tmp_name[1]
              else:
                ch, py = tmp_name, None

              try:
                py = py.replace(')','')
              except:
                py = None

              try:
                g = tmp[1].replace(' name', '')
              except:
                g = None

            else:
              print(f"The div was not found ({name}).")
              ch, py, g = name, None, None

            # save to cache
            if py is not None and g is not None:
              self.save_to_cache(name, ch, py, g)

            return (ch, py, g)

      except Exception as ex:
        print(ex)
      
      tries -= 1

      if tries <= 0:
        print(f"Failed to retrieve content. Too many tries. {name}")
        print(f"Responses: {responses}")
        return (name, None, None)
