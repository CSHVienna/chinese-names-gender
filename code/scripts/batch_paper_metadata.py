# export PYTHONPATH="${PYTHONPATH}:../"
# export PATH=$PATH:/usr/local/bin/chromedriver

## NOTES: MagicCNKI, CrossRef, SemanticWeb do not work (search results are empty)

import argparse
import os
import time

import numpy as np
import pandas as pd
from libs.cnki import CNKI

TITLES_SAMPLE = [
    "亚皮秒光整流效应",
    "THz电磁场的新型电光探测技术及应用",
    "THz技术的应用及展望",
    "使用太赫兹技术研究航天飞机失事的原因",
    "双光子激发钾5d2DJ态的光波混频和参量过程",
    "半导体激光器在光谱研究中的应用",
]
HEADER = 0
CHUNK = [(1, 14), (17, 5), (24, 7), (33, 6), (41, 7), (50, 12), (64, 10), (76, 7), (85, 7), (94, 9)]


def crawl_chunk(df, output_dir, cache_dir, chunk_id=1, prefix=''):
    df_publications = pd.DataFrame()
    df_citations = pd.DataFrame()
    df_references = pd.DataFrame()

    for id, row in df.iterrows():
        # paper
        paper_id = id + 1
        title = row.title
        dict_paper = None

        try:
            var = CNKI(cache_dir=cache_dir)
            var.setup()
            dict_paper = var.search_paper_by_title(title)
            tmp = pd.DataFrame(dict_paper, index=[0])
            tmp.loc[:, 'main_paper_id'] = paper_id
            tmp.loc[:, 'chunk_id'] = chunk_id
            tmp.rename(columns={c: c.lower().replace(' ', '_') for c in tmp.columns}, inplace=True)
            tmp.rename(columns={'id': 'result_rank'}, inplace=True, errors='ignore')
            # tmp.loc[:,'original_title'] = title
            df_publications = pd.concat([df_publications, tmp], ignore_index=True)
            df_publications.drop(columns=['Id', 'Options'], inplace=True, errors='ignore')
            df_publications.loc[:, 'cites'] = df_publications.cites.apply(lambda v: 0 if v in [None, '', ' '] else int(v))
            df_publications.loc[:, 'references_count'] = 0
        except Exception as ex:
            print(ex)
            pass

        # citations (other -newer- papers citing the main papers from my list)
        try:
            if dict_paper is not None:
                var = CNKI(cache_dir=cache_dir)
                var.setup()
                list_dict_citations = var.get_citations(dict_paper)
                if list_dict_citations is not None and len(list_dict_citations) > 0:
                    tmp = pd.DataFrame(list_dict_citations)
                    tmp.loc[:, 'main_paper_id'] = paper_id
                    tmp.loc[:, 'chunk_id'] = chunk_id
                    # tmp.loc[:, 'original_title'] = title
                    df_citations = pd.concat([df_citations, tmp], ignore_index=True)


        except Exception as ex:
            print(ex)
            pass

        # references (other -older- papers the main papers from my list cite)
        try:
            if dict_paper is not None:
                var = CNKI(cache_dir=cache_dir)
                var.setup()
                list_dict_references = var.get_references(dict_paper)
                if list_dict_references is not None and len(list_dict_references) > 0:
                    tmp = pd.DataFrame(list_dict_references)
                    tmp.loc[:, 'main_paper_id'] = paper_id
                    tmp.loc[:, 'chunk_id'] = chunk_id
                    # tmp.loc[:, 'original_title'] = title
                    df_references = pd.concat([df_references, tmp], ignore_index=True)

        except Exception as ex:
            print(ex)
            pass

    # update # citations
    df_publications.loc[:, 'citations_count'] = df_publications.apply(lambda row: df_citations.query("chunk_id==@row.chunk_id and main_paper_id==@row.main_paper_id").shape[0], axis=1).astype(int)

    # update # references
    df_publications.loc[:, 'references_count'] = df_publications.apply(lambda row: df_references.query("chunk_id==@row.chunk_id and main_paper_id==@row.main_paper_id").shape[0], axis=1).astype(int)

    # outputs
    print(f"======= Summary =======")

    if df_publications.shape[0] > 0:
        print(f"\nPublications:")
        print_full(df_publications.head(2))
        print(f"- Publications: {df_publications.shape} (of {df.shape[0]})")
        df_publications.set_index(['chunk_id', 'main_paper_id']).to_csv(
            os.path.join(output_dir, f'{prefix}_publications.csv'), index=True)

    if df_citations.shape[0] > 0:
        print(f"\nCitations:")
        print_full(df_citations.head(2))
        print(f"- Citations: {df_citations.shape} (of {df_publications.cites.sum()})")
        df_citations.set_index(['chunk_id', 'main_paper_id', 'order']).to_csv(
            os.path.join(output_dir, f'{prefix}_citations.csv'), index=True)
    else:
        print(f"\nNo Citations.")

    if df_references.shape[0] > 0:
        print(f"\nReferences:")
        print_full(df_references.head(2))
        print(f"- References: {df_references.shape}")
        df_references.set_index(['chunk_id', 'main_paper_id', 'order']).to_csv(
            os.path.join(output_dir, f'{prefix}_references.csv'), index=True)
    else:
        print(f"\nNo References.")

def print_full(df):
    from tabulate import tabulate
    print(tabulate(df.drop(columns=['dbname','source','scode','sdb','jounnal','result_rank','database','downloads','order','filename','dbcode'], errors='ignore'), headers='keys', tablefmt='psql'))

def test_cnki(output_dir, cache_dir):
    df = pd.DataFrame({'title': TITLES_SAMPLE}, index=range(len(TITLES_SAMPLE)))
    crawl_chunk(df, output_dir, cache_dir, chunk_id=1, prefix='test')


def sample_all_chunks(xls_fn, output_dir, cache_dir):
    chunk_id = 1
    for start, nrows in CHUNK:
        df = pd.read_excel(xls_fn, index_col=None, header=HEADER, skiprows=np.arange(1, start), nrows=nrows)
        df.loc[:, 'person_id'] = chunk_id
        crawl_chunk(df, output_dir, cache_dir, chunk_id=chunk_id, prefix=f'chunk{chunk_id}')
        chunk_id += 1
        break
    return


def sample_one_chunk(xls_fn, chunk_id, output_dir, cache_dir):
    start, nrows = CHUNK[chunk_id - 1]
    df = pd.read_excel(xls_fn, index_col=None, header=HEADER, skiprows=np.arange(1, start), nrows=nrows)
    df.loc[:, 'person_id'] = chunk_id
    crawl_chunk(df, output_dir, cache_dir, chunk_id=chunk_id, prefix=f'chunk{chunk_id}')


if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description='Crawls CNKI to extract citation papers.')

    # Add arguments
    parser.add_argument('--xls_fn', help='name_citation_test.xlsx', type=str, required=True)
    parser.add_argument('--output_dir', help='../../results/CNKI', type=str, required=True)
    parser.add_argument('--cache_dir', help='../../cache', type=str, required=True)
    parser.add_argument('--chunk_id', help='Id from 1 to 10', type=int, required=True, default=1)
    parser.add_argument('--test', help='Simple test or not', action='store_true')

    # Parse the arguments
    args = parser.parse_args()
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))

    start_time = time.time()
    if args.test:
        print("== SIMPLE TEST ==")
        test_cnki(output_dir=args.output_dir, cache_dir=args.cache_dir)
        pass
    else:
        sample_one_chunk(xls_fn=args.xls_fn, chunk_id=args.chunk_id, output_dir=args.output_dir, cache_dir=args.cache_dir)
        pass
    print(f"--- {time.time() - start_time} seconds ({args.chunk_id}) ---")

