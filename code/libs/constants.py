from numpy import nan

MAX_RESULTS_PER_PAGE = 200
MAX_RESULTS_PER_PAGE_TEST = 20
MAX_ITEMS_IN_FILTER = 40
PRECISION = 4
STRING_SEPARATOR = '||'
COMMA_SEPARATOR = ', '
MIN_SCORE = 0.5
NONE = ['', ' ', None, 'nan', 'None', nan, 'NaN']
LAST_10_YEARS = 11

FN_PUBLICATIONS = 'publications.csv'
FN_CITATIONS = 'citations.csv'
FN_AUTHORS = 'authors.csv'
FN_AFFILIATIONS = 'institutions.csv'

FN_CHUNKS = 'chunks_gender'
FN_FULL_NAMES = 'CNKI_uniqueChineseNames.csv'
FN_GIVEN_NAMES = 'CNKI_uniqueChineseGivenNames.csv'
FN_CH_GIVEN_NAMES = 'CNKI_uniqueChineseGivenNames_only.csv'
FN_EN_GIVEN_NAMES = 'CNKI_uniqueEnglishGivenNames_only.csv'
FN_FULL_NAMES_GENDER = 'CNKI_uniqueChineseNames_gender.csv'
FN_CH_GIVEN_NAMES_GENDER_1CH = 'CNKI_uniqueChineseCharacter_gender.csv'
FN_CH_GIVEN_NAMES_GENDER_2CH = 'CNKI_uniqueChineseGivenNames_gender.csv'
FN_CH_GIVEN_NAMES_GENDER = 'CNKI_uniqueChineseAllGivenNames_gender.csv'
FN_EN_GIVEN_NAMES_GENDER = 'CNKI_uniqueEnglishGivenNames_gender.csv'
FN_GIVEN_NAMES_1CH = 'CNKI_uniqueChineseCharacter.csv'
FN_GIVEN_NAMES_2CH = 'CNKI_uniqueChineseGivenNames_CNKIpop.csv'
FN_ANNOTATIONS = "ground_truth.csv"

GENDER_BINARY_ORDER = ['female', 'male'] # TODO: import from categorical and use MALE and FEMALE