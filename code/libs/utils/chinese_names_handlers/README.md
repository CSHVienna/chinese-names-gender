# Chinese Names Handlers 

Author: Liuhuaying yang

The Chinese language is character-based, including personal names. While surnames are inherited, parents can choose one or two (rarely three) characters for their children’s given names based on factors like meaning, sound, and family tradition.

## Detect Chinese names

### Detect from Pinyin syllables
When presenting Chinese names in western contexts, these names have been transliterated to Hanyu Pinyin, which represents Mandarin pronunciation, rather than their original characters. 
For example, in Cixin Liu, "Liu" (刘) is the surname, and "Cixin" is the chosen given name, made up of the characters "Ci" (慈) and "Xin" (欣). When written in Pinyin, these characters are represented as syllables that together form the name "Cixin."

**Detect whether a English name is a Chinese name**

```
from pinyinnames_handler import is_pinyin_givenname, is_pinyin_familyname, is_pinyin_name

# Detect a given name
print(is_pinyin_givenname("mingrong"))      
# Example output: 2 (it can be a Chinese given name with two characters)

print(is_pinyin_givenname("liuhuaying", allow_3chars=True))  
# Example output: 3

print(is_pinyin_givenname("xyz"))           
# Example output: False (not a valid Pinyin name)

# Detect a family name
print(is_pinyin_familyname("yang"))          # Example output: True
print(is_pinyin_familyname("dongfang"))      # Example output: True (compound family name)
print(is_pinyin_familyname("abc"))       # Example output: False

# Detect a full name
print(is_pinyin_name('yang', 'liuhuaying', allow_3chars=True))  
# Example output: 3

print(is_pinyin_name('yang', 'liuhuaying'))  
# Example output: False
```

**Possible Returns**

| Function              | Return  | Meaning                                                |
| --------------------- | ------- | ------------------------------------------------------ |
| `is_pinyin_givenname` | `False` | The name cannot be a valid Chinese given name.  |
|                       | `1`     | The name can be a valid single-character given name.        |
|                       | `2`     | The name can be a valid two-character given name.       |
|                       | `3`     | The name can be a valid three-character given name.                       |
|                       | `4`     | The name can be a valid given name (mixed length of characters).                       |
| `is_pinyin_familyname`      | `False` | The name cannot be a valid Chinese family name.                  |
|                       | `True` | The name can be a valid Chinese family name. |
| `is_pinyin_name`      | `False` | The full name cannot be a valid Chinese name.                  |
|                       | `1,2,3,4` | Same as `is_pinyin_givenname` if family name is valid. |


### Detect from Chinese characters
**Return the valid Chinese names**

Valid criteria: the name must use only characters from the commonly used character list, and the given name must contain no more than 2 characters.

```
from characternames_handler import split_names, pinyinname

print(split_names("诸葛亮,任茜,杨柳桦樱"))
# Example output: ['诸葛 亮', '任 茜'] (a list of valid Chinese character names in format of 'surname givenname')
```

## Translate Chinese names
Some Chinese characters have multiple pronunciations or special pronunciations when used in names. Here, only the most common pronunciation for names is considered.

**Return the pinyin (without tones) for Chinese names**
```
from characternames_handler import split_names, pinyinname

print(pinyinname(['诸葛 亮', '任 茜']))
# Example output: ['zhuge liang', 'ren xi'] (a list of pinyin names in format of 'surname givenname')
```
