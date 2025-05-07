import os
import datetime
import json
import hashlib
import numpy as np
import pandas as pd
import requests

from constants import *

class OpenAlex:
    ORCID_URL_PREFIX = 'https://orcid.org/'
    OA_URL_PREFIX = 'https://openalex.org/'
    OA_API_URI = 'https://api.openalex.org'
    GSCHOLAR_URL_PREFIX = 'https://scholar.google.com/citations?user='
    ROR_URL_PREFIX = 'https://ror.org/'

    OK = 0
    GSCHOLAR_WRONG_URL = 1
    ORCID_WRONG_URL = 2
    ORCID_OK = 3
    GSCHOLAR_OK = 4
    NAMES_OK = 5

    def __init__(self, id=None, name=None, lastname=None, scholar_url=None, orcid=None, emails=None, cache_dir=None):
        self.id = id
        self.name = name
        self.lastname = lastname
        self.scholar_url = scholar_url
        self.orcid = None if not orcid else orcid.replace(OpenAlex.ORCID_URL_PREFIX, '')
        self.emails = emails
        self.cache_dir = cache_dir

    ############################################################
    # API CALL
    ############################################################

    def get_email(self):
        return np.random.choice(self.emails, 1)[0] if self.emails else None

    @staticmethod
    def get_uri(entity, filter, email, cursor='*'):
        # entity: authors, works, institutions
        url = f"{OpenAlex.OA_API_URI}/{entity}?filter={filter}&per-page={MAX_RESULTS_PER_PAGE}&cursor={cursor}&mailto={email}"
        return url

    @staticmethod
    def get_cache_fn(api_url, cache_dir):
        entity = api_url.split(f"{OpenAlex.OA_API_URI}/")[-1].split('?')[0]
        filter = api_url.split('filter=')[-1].split('&')[0]
        cursor = api_url.split('cursor=')[-1].split('&')[0]
        page = api_url.split('per-page=')[-1].split('&')[0]

        fn = f"{filter}_{cursor}_{page}".replace('.','_').replace(" ",'_').replace('*','_')
        fn = f"{hashlib.sha512(fn.encode()).hexdigest()}.json"

        path = os.path.join(cache_dir, entity)
        os.makedirs(path, exist_ok=True)

        cache_fn = os.path.join(path, fn)
        return cache_fn

    @staticmethod
    def cache(data, api_url, cache_dir):
        fn = OpenAlex.get_cache_fn(api_url, cache_dir)
        try:
            with open(fn, 'w') as f:
                json.dump(data, f)
        except Exception as ex:
            print(ex)

    @staticmethod
    def load_from_cache(api_url, cache_dir):
        fn = OpenAlex.get_cache_fn(api_url, cache_dir)
        try:
            with open(fn, 'r') as f:
                data = json.load(f)
                return data
        except Exception as ex:
            # print(ex)
            pass

        return None

    @staticmethod
    def call_api(api_url, cache_dir):

        data = OpenAlex.load_from_cache(api_url, cache_dir)
        if data:
            print('=== loaded from cache ===')
            return data

        print('=== queried API ===')
        try:
            response = requests.get(api_url)
            response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes
            data = response.json()  # Parse the response as JSON
            OpenAlex.cache(data=data, api_url=api_url, cache_dir=cache_dir)
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return None

    ############################################################
    # SEARCH
    ############################################################

    # def search(self):
    #     status = self.OK
    #     results = ''

    #     if self.name and self.lastname:
    #         status = self.NAMES_OK
    #         results = self._search_by_name()
    #         return results, status

    #     if self.scholar_url:
    #         if not self.scholar_url.startswith(OpenAlex.GSCHOLAR_URL_PREFIX):
    #             status = self.GSCHOLAR_WRONG_URL
    #         else:
    #             status = self.GSCHOLAR_OK
    #             results = 'Under construction...'
    #         return results, status

    #     if self.orcid:
    #         if self.orcid.startswith('http') and not self.orcid.startswith(OpenAlex.ORCID_URL_PREFIX):
    #             status = self.ORCID_WRONG_URL
    #         else:
    #             status = self.ORCID_OK
    #             results = self._search_by_orcid()
    #         return results, status

    #     return results, status

    # def _search_by_name(self, cursor='*'):

    #     results = None
    #     while cursor:
    #         filter = f"display_name.search:{self.name} {self.lastname}"
    #         email = self.get_email()
    #         url = OpenAlex.get_uri(entity='authors', filter=filter, cursor=cursor, email=email)

    #         tmp = OpenAlex.call_api(url, self.cache_dir)

    #         cursor = None if not tmp else tmp['meta']['next_cursor']
    #         if results is None:
    #             results = tmp
    #         else:
    #             results['results'].extend(tmp['results'])

    #     return results

    # def _search_by_orcid(self):
    #     filter = f"orcid:{self.orcid}"
    #     email = self.get_email()
    #     url = OpenAlex.get_uri(entity='authors', filter=filter, email=email)
    #     results = OpenAlex.call_api(url, self.cache_dir)
    #     return results

    # @staticmethod
    # def parse_search_results(results):
    #     html = "<h3>Results</h3>"
    #     meta = results['meta']
    #     one = meta['count'] == 1
    #     html = html + f"<p>{meta['count']} author{'' if one else 's'} match{'es' if one else ''} your query.</p>"

    #     html = html + "<table id='table-results'><thead><tr><th>#</th><th>Name</th><th>Hint</th><th>Publications</th><th>Citations</th><th>h-index</th><th>i10-index</th><th>Last known institution</th><th>ORCID</th><th>Download</th><th>Visualize</th></tr></thead><tbody>"
    #     counter = 0
    #     for obj in results['results']:
    #         counter += 1
    #         id = obj['id'].replace(OpenAlex.OA_URL_PREFIX, '')
    #         hints = [obj2['display_name'] for obj2 in obj['x_concepts'] if obj2['score'] > 0]
    #         # hints = np.random.choice(hints, min(3, len(hints)), replace=False)
    #         hints = f"{COMMA_SEPARATOR}".join(hints[:min(3, len(hints))])

    #         html = html + "<tr>"
    #         html = html + f"<td>{counter}</td>"
    #         html = html + f"<td>{obj['display_name']}</td>"
    #         html = html + f"<td>{hints}</td>"
    #         html = html + f"<td>{obj['works_count']}</td>"
    #         html = html + f"<td>{obj['cited_by_count']}</td>"
    #         html = html + f"<td>{obj['summary_stats']['h_index']}</td>"
    #         html = html + f"<td>{obj['summary_stats']['i10_index']}</td>"
    #         html = html + f"<td>{'' if not obj['last_known_institutions'] else f'{COMMA_SEPARATOR}'.join([lki['display_name'] for lki in obj['last_known_institutions']])}</td>"
    #         if obj['orcid']:
    #             html = html + f"<td><a href={obj['orcid']} target='_blank'>{obj['orcid'].replace(OpenAlex.ORCID_URL_PREFIX, '')}</a></td>"
    #         else:
    #             html = html + f"<td></td>"
    #         html = html + f"<td style='text-align: center'><button id='button_d{id}' class='download-network button-network' onclick='download_network(\"{id}\")' title='Download' alt='Download'></button></td>" \
    #                       f"<td style='text-align: center'><button id='button_s{id}' class='show-network button-network' onclick='plot_network(\"{id}\")' title='Visualize' alt='Visualize'></button></td>"
    #         html = html + "</tr>"
    #     html = html + "</tbody></table>"
    #     return html

    ############################################################
    # DOWNLOAD
    ############################################################

    def get_publication(self, title):
        publications = pd.DataFrame()
        citations = pd.DataFrame()

        email = self.get_email()
        cursor = '*'
        while cursor:
            filter = f"title.search:{title}"
            url = OpenAlex.get_uri(entity='works', filter=filter, cursor=cursor, email=email)

            results = OpenAlex.call_api(url, self.cache_dir)
            cursor = results['meta']['next_cursor']

            for obj in results['results']:
                # publications
                df = OpenAlex.get_publication_metadata(obj)
                publications = pd.concat([publication, df], ignore_index=True)

                # citations
                df = OpenAlex.get_citations_metadata(obj, email,
                                                     cache_dir=self.cache_dir)
                citations = pd.concat([citations, df], ignore_index=True)
                
    
    @staticmethod
    def get_publication_metadata(obj):
        authors = [a['author']['id'].replace(OpenAlex.OA_URL_PREFIX, '') for a in obj['authorships']]
        countries = f"{COMMA_SEPARATOR}".join([c for a in obj['authorships'] for c in a['countries']])
        work_id = obj['id'].replace(OpenAlex.OA_URL_PREFIX, '')
        # apos = [a['author_position'] for a in obj['authorships'] if author_id in a['author']['id']]
        # isc = [a['is_corresponding'] for a in obj['authorships'] if author_id in a['author']['id']]
        sdgs = [sdg['id'].split('/')[-1] for sdg in obj['sustainable_development_goals']]
        sdgs_scores = [str(sdg['score']) for sdg in obj['sustainable_development_goals']]
        topics = set([oc['display_name'] for oc in obj['topics'] if oc['score'] > MIN_SCORE])
        keywords = set([kw['keyword'] for kw in obj['keywords'] if kw['score'] > MIN_SCORE])
        concepts = set([co['display_name'] for co in obj['concepts'] if co['score'] > MIN_SCORE])
        corresponding_author_ids = f"{COMMA_SEPARATOR}".join([ca.replace(OpenAlex.OA_URL_PREFIX, '').strip() for ca in obj['corresponding_author_ids']])

        tmp = {'id': work_id,
               'doi': obj['doi'],
               'title': obj['title'],
               'language': obj['language'],
               'publication_date': obj['publication_date'],
               'type': obj['type'] if obj['type'] else None,
               'type_crossref': obj['type_crossref'] if obj['type_crossref'] else None,
               'publisher': obj['primary_location']['source']['display_name'] if obj['primary_location'] and obj['primary_location']['source'] else None,
               'is_published': obj['primary_location']['is_published'] if obj['primary_location'] else None,
               'apc_paid_value': obj['apc_paid']['value_usd'] if 'apc_paid' in obj and obj['apc_paid'] and 'value_usd' in obj['apc_paid'] else None,
               'open_access': obj['open_access']['is_oa'] if obj['open_access'] and 'is_oa' in obj['open_access'] else None,
               'primary_topic_display_name': obj['primary_topic']['display_name'] if obj['primary_topic'] else None,
               'primary_topic_field_display_name': obj['primary_topic']['field']['display_name'] if obj['primary_topic'] and 'field' in obj['primary_topic'] else None,
               'primary_topic_subfield_display_name': obj['primary_topic']['subfield']['display_name'] if obj['primary_topic'] and 'subfield' in obj['primary_topic']  else None,
               'primary_topic_domain_display_name': obj['primary_topic']['domain']['display_name'] if obj['primary_topic'] and 'domain' in obj['primary_topic']  else None,
               'num_authors': len(authors),
               'authors': f"{COMMA_SEPARATOR}".join(authors),
               'corresponding_author_ids': corresponding_author_ids,
              #  'author_position': apos[0] if len(apos) > 0 else None,
              #  'is_corresponding': isc[0] if len(isc) > 0 else None,
               'referenced_works_count': obj['referenced_works_count'],
               'received_citations': obj['cited_by_count'],
               'countries': countries,
               'countries_distinct_count': obj['countries_distinct_count'],
               'institutions_distinct_count': obj['institutions_distinct_count'],
               'online_locations_count': obj['locations_count'],
               'is_retracted': obj['is_retracted'],
               'is_paratext': obj['is_paratext'],
               'sdgs': f'{COMMA_SEPARATOR}'.join(sdgs),
               'sdg_score': f'{COMMA_SEPARATOR}'.join(sdgs_scores),
               'topics': f'{STRING_SEPARATOR}'.join(topics),
               'keywords': f'{STRING_SEPARATOR}'.join(keywords),
               'concepts': f'{STRING_SEPARATOR}'.join(concepts),}

        df = pd.DataFrame(tmp, index=[0])
        return df

    @staticmethod
    def get_citations_metadata(obj, email, cache_dir):
        work_id = obj['id'].replace(OpenAlex.OA_URL_PREFIX, '')
        in_cit = OpenAlex.get_citations(work_id, email=email, incoming=True, cache_dir=cache_dir)
        out_cit = OpenAlex.get_citations(work_id, email=email, incoming=False, cache_dir=cache_dir)
        all_citations = pd.concat([in_cit, out_cit], ignore_index=True)
        return all_citations

    @staticmethod
    def get_citations(work_id, email, cache_dir, incoming=True):
        # incoming citations: works that cite the given work
        # outgoing citations: works cited by the given work (works found in the given work's referenced_works section.)

        citations = pd.DataFrame()
        kind = 'cites' if incoming else 'cited_by'
        filter = f"{kind}:{work_id}"
        cursor = '*'
        while cursor:
            url = OpenAlex.get_uri(entity='works', filter=filter, cursor=cursor, email=email)
            results = OpenAlex.call_api(url, cache_dir)
            cursor = results['meta']['next_cursor'] if results and 'meta' in results and 'next_cursor' in results['meta'] else None

            for obj in results['results']:
                authors = [a['author']['id'].replace(OpenAlex.OA_URL_PREFIX, '') for a in obj['authorships']]
                countries = f"{COMMA_SEPARATOR}".join(sorted(set(obj['authorships']['countries']))) if 'countries' in obj['authorships'] and obj['authorships']['countries'] else None
                sdgs = [sdg['id'].split('/')[-1] for sdg in obj['sustainable_development_goals']]
                sdgs_scores = [str(sdg['score']) for sdg in obj['sustainable_development_goals']]
                topics = set([oc['display_name'] for oc in obj['topics'] if oc['score'] > MIN_SCORE])
                keywords = set([kw['keyword'] for kw in obj['keywords'] if kw['score'] > MIN_SCORE])
                concepts = set([co['display_name'] for co in obj['concepts'] if co['score'] > MIN_SCORE])
                # topic_similarity = helper.jaccard_similarity(main_work_topics, topics, PRECISION)
                # keyword_similarity = helper.jaccard_similarity(main_work_keywords, keywords, PRECISION)
                # concept_similarity = helper.jaccard_similarity(main_work_concepts, concepts, PRECISION)
                # sdg_similarity = helper.jaccard_similarity(main_work_sdgs, sdgs, PRECISION)

                tmp = {'id': obj['id'].replace(OpenAlex.OA_URL_PREFIX, ''),
                       'doi': obj['doi'], # NEW
                       'title': obj['title'],
                       'language': obj['language'],
                       'publication_date': obj['publication_date'], # NEW
                       'type': obj['type'] if obj['type'] else None,
                       'type_crossref': obj['type_crossref'] if obj['type_crossref'] else None, # NEW
                       'publisher': obj['primary_location']['source']['display_name'] if obj['primary_location'] and obj['primary_location']['source'] else None, # NEW
                       'is_published': obj['primary_location']['is_published'] if obj['primary_location'] else None, # NEW
                       'apc_paid_value': obj['apc_paid']['value_usd'] if 'apc_paid' in obj and obj['apc_paid'] and 'value_usd' in obj['apc_paid'] else None, # NEW
                       'open_access': obj['open_access']['is_oa'] if obj['open_access'] and 'is_oa' in obj['open_access'] else None, # NEW
                       'primary_topic_display_name': obj['primary_topic']['display_name'] if obj['primary_topic'] else None, # NEW
                       'primary_topic_field_display_name': obj['primary_topic']['field']['display_name'] if obj['primary_topic'] and 'field' in obj['primary_topic'] else None, # NEW
                       'primary_topic_subfield_display_name': obj['primary_topic']['subfield']['display_name'] if obj['primary_topic'] and 'subfield' in obj['primary_topic'] else None, # NEW
                       'primary_topic_domain_display_name': obj['primary_topic']['domain']['display_name'] if obj['primary_topic'] and 'domain' in obj['primary_topic'] else None, # NEW
                       'num_authors': len(authors),
                       'authors': f"{COMMA_SEPARATOR}".join(authors),
                       'referenced_works_count': obj['referenced_works_count'],
                       'received_citations': obj['cited_by_count'],
                       'countries': countries, # NEW
                       'countries_distinct_count': obj['countries_distinct_count'],
                       'institutions_distinct_count': obj['institutions_distinct_count'],
                       'online_locations_count': obj['locations_count'],
                       'is_retracted': obj['is_retracted'],
                       'is_paratext': obj['is_paratext'],
                       'sdgs': f'{COMMA_SEPARATOR}'.join(sdgs),
                       'sdg_score': f'{COMMA_SEPARATOR}'.join(sdgs_scores), # NEW
                       'topics': f'{STRING_SEPARATOR}'.join(topics), # NEW
                       'keywords': f'{STRING_SEPARATOR}'.join(keywords), # NEW
                       'concepts': f'{STRING_SEPARATOR}'.join(concepts), # NEW
                       'reference_kind': kind,
                       'reference_id': work_id,
                      #  'topic_similarity': topic_similarity,
                      #  'keyword_similarity': keyword_similarity, # NEW
                      #  'concept_similarity': concept_similarity, # NEW
                      #  'sdg_similarity': sdg_similarity, # NEW
                       }

                df = pd.DataFrame(tmp, index=[0])
                citations = pd.concat([citations, df], ignore_index=True)

        return citations
