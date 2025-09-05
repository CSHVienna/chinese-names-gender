# Chinese Names Handlers 

**Notebooks for handling Chinese names in Pinyin or Chinese**

## Detect Chinese names

### Detect from Pinyin syllables

```
from pinyinnames_handler import is_pinyin_givenname, is_pinyin_familyname, is_pinyin_name

# Detect a given name
print(is_pinyin_givenname("mingrong")) 
print(is_pinyin_givenname("liuhuaying", 3)) # allows 3 characters, default is 2

# Detect a family name
print(is_pinyin_familyname("yang"))
print(is_pinyin_familyname("dongfang")) # detects compound family names

# Detect a full name
is_pinyin_name('yang','liuhuaying',3) # family name, given name, max given name length
```

### Detect from Chinese characters
