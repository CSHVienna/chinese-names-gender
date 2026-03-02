# function to split the whole givenname/firstname in pinyin(Latin script/English) into pinyin by characters.
def split_pinyin(pinyin):
    pinyin_syllables = {
        "yi","er","shi","ding","zheng","chang","an","han","qi","bu","ba","ren","ji","jiu","diao","liao","dao","li","nai","you","san","gan","yu","gong","tu","cai","xia","cun","da","dai","tai","wan","mo","shang","xiao","kou","shan","jin","qian","chuan","ge","xi","yao","fan","guang","men","ya","zhi","si","zi","wei","ye","nv","ru","fei","ma","xiang","feng","wang","kai","jing","tian","fu","yuan","wu","yun","zhuan","za","zha","mu","ting","fou","qu","ou","che","ju","tun","zhun","bi","wa","shao","ri","zhong","bei","gang","shui","jian","xian","niu","mao","sheng","zhang","shen","pu","hua","chou","qiu","reng","jie","cong","zong","lun","xiong","fen","cang","yue","dan","gou","liu","lu","wen","kang","fang","huo","dou","hu","xin","yin","kong","deng","quan","shuang","shu","kan","qiao","hui","ai","gu","ben","zhu","ke","bing","zuo","hang","long","ping","dong","qia","ka","zhan","shuai","gui","qie","xie","jia","dian","hao","yang","jiao","tao","ran","he","bai","zai","ta","ling","yong","le","wai","niao","bao","xuan","lan","tou","ning","rang","xun","ni","min","hong","chu","zhao","pi","bian","fa","bang","xing","rong","kao","tuo","lao","di","mang","po","piao","guo","chen","xu","duo","lie","cheng","ga","mai","zhen","dang","zao","chong","tuan","lv","tong","sui","ze","rou","nian","xiu","yan","bin","ang","xue","hou","zhou","kuai","chuang","ming","se","zhuang","qing","zhai","ci","chan","guan","mi","jiang","chi","tang","shou","jun","nong","she","na","nei","sun","pei","huan","tan","zhe","keng","que","qin","su","du","sha","geng","hai","lai","lian","kuang","nan","kun","chui","dun","bie","wo","mei","ti","bo","fo","lin","jue","tiao","ying","ku","leng","run","dui","rui","can","zhuo","tui","huai","song","liang","ceng","gai","a","e","miao","nu","chun","biao","la","ruo","re","peng","ban","zhuai","ken","tie","luo","pa","die","guai","suo","pin","pang","nao","juan","pao","lang","meng","cen","zu","zou","nuo","cao","cha","huang","ce","mian","pan","sai","ha","chao","duan","shun","zhui","pen","rao","chai","lei","lou","shuo","sa","hun","heng","nie","shuan","gen","beng","en","zuan","te","weng","gao","kuan","neng","sang","cuo","pai","man","lve","cui","de","dei","cou","cu","mou","chuo","zeng","qiong","chuai","kui","sen","hei","ao","kuo","zun","qiang","nuan","teng","gun","miu","qun","seng","nen","zui","pian","zan","zang","mie","ne","ruan","jiong","luan","gua"
    }
    
    def helper(start):
        if start == len(pinyin):
            return [[]]  # Base case: reached the end of the string

        results = []
        for end in range(start + 1, len(pinyin) + 1):
            syllable = pinyin[start:end]

            if syllable in pinyin_syllables:
                rest = helper(end)  # Recursively find the rest of the syllables
                for r in rest:
                    combined_result = [syllable] + r
                    if len(combined_result) <= 3:  # limit to max three items
                        results.append(combined_result)

        results.sort(key=len)  # Sort results by length
        return results

    return helper(0)

# function to detect if an English givenname a pinyin givenname.
def is_pinyin_givenname(name, allow_3chars=False):
    
    results = split_pinyin(name.lower())  # normalize to lowercase
    
    if not results:  # no valid splits
        return False

    # Determine max allowed length based on flag
    max_len = 3 if allow_3chars else 2
        
    # Filter splits that satisfy max_len
    valid_splits = [r for r in results if len(r) <= max_len]
    if not valid_splits:
        return False
    
    # Check lengths
    lengths = set(len(r) for r in valid_splits)
    if len(lengths) == 1:
        return lengths.pop()  # return the single length
    else:
        return 4  # mixed lengths



# function to detect if an English family name a pinyin familyname.
compound_familynames = [
    {"name": "令狐", "english": "linghu"},
    {"name": "尉迟", "english": "yuchi"},
    {"name": "澹台", "english": "tantai"},
    {"name": "皇甫", "english": "huangfu"},
    {"name": "长孙", "english": "zhangsun"},
    {"name": "宰父", "english": "zaifu"},
    {"name": "亓官", "english": "qiguan"},
    {"name": "毌丘", "english": "guanqiu"},
    {"name": "欧阳", "english": "ouyang"},
    {"name": "上官", "english": "shangguan"},
    {"name": "诸葛", "english": "zhuge"},
    {"name": "司徒", "english": "situ"},
    {"name": "司马", "english": "sima"},
    {"name": "申屠", "english": "shentu"},
    {"name": "夏侯", "english": "xiahou"},
    {"name": "贺兰", "english": "helan"},
    {"name": "完颜", "english": "wanyan"},
    {"name": "慕容", "english": "murong"},
    {"name": "鲜于", "english": "xianyu"},
    {"name": "闻人", "english": "wenren"},
    {"name": "颛孙", "english": "zhuansun"},
    {"name": "东方", "english": "dongfang"},
]
compound_familyname_set = {item["english"].lower() for item in compound_familynames}

def is_pinyin_familyname(name):
    name = name.lower()
    
    # Condition 1: compound surname
    if name in compound_familyname_set:
        return True

    # Condition 2: single syllable from split_pinyin
    results = split_pinyin(name)
    if any(len(r) == 1 for r in results):
        return True
    
    return False
    

def is_pinyin_name(familyname,givenname, allow_3chars=False):
    # Check family name
    if not is_pinyin_familyname(familyname):
        return False
    
    # Check given name (detailed)
    return is_pinyin_givenname(givenname, allow_3chars=allow_3chars)
    
