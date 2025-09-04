import pandas as pd
import re
import string
import pickle
import csv
from tqdm import tqdm


def clean_list(data, dict_data):
    """
    Special cleaning for specific cases of author lists.
    Handles hard-coded problematic cases by replacing unwanted characters.

    :param data: list of author names
    :param dict_data: dictionary mapping authors to departments
    :return: (cleaned author list, updated dictionary)
    """
    if data == ['ShosatkuTanaka', '101组']:
        updated_dict_data = {key.replace('101', ''): value for key, value in dict_data.items()}
        return ['ShosatkuTanaka', '组'], updated_dict_data
    elif data == ['\u3000清泉', '朱俊', '李萍']:
        updated_dict_data = {key.replace('\u3000', ''): value for key, value in dict_data.items()}
        return ['朱俊', '李萍', '清泉'], updated_dict_data
    else:
        return data, dict_data


def clean_name_text(text):
    """
    Clean unwanted characters from names using regex.
    Removes leading/trailing numbers, letters, punctuation, and spaces.

    :param text: input author name
    :return: cleaned author name
    """
    cleaned_text = re.sub(
        r'^[A-Za-z0-9' + re.escape(string.punctuation) + r'\s]+|[A-Za-z0-9' + re.escape(string.punctuation) + r'\s]+$', 
        '', 
        text
    )
    return cleaned_text


def diss_depart(pid, authors_list, authors_with_index, department_info, notmatch):
    """
    Assign departments to authors by parsing author index and department text.

    :param pid: paper id
    :param authors_list: list of author names
    :param authors_with_index: string with authors and indices
    :param department_info: string containing department information
    :param notmatch: set to record paper IDs with mismatched department mapping
    :return: dictionary mapping authors to department(s)
    """
    department_info = department_info.rstrip(',')

    # Case 1: Authors contain numeric indices (e.g., Author1, Author2)
    if '1' in authors_with_index and '1' in department_info:
        authors_with_index_list_correct = re.findall(r'([^\d]+)(\d+(?:,\d+)*)', authors_with_index)
        department_info_corrected_matches = re.findall(r'(\d+)\. (.*?)(?=\d+\. |$)', department_info, re.DOTALL)
        departments_correct = {match[0]: match[1].strip() for match in department_info_corrected_matches}

        author_department_dict_update = {}
        for author, indices_str in authors_with_index_list_correct:
            indices = re.findall(r'\d+', indices_str)
            dep_names = [departments_correct[index] for index in indices]
            author_department_dict_update[author.replace(" ", "")] = dep_names

    # Case 2: Direct one-to-one mapping with commas
    elif len(department_info.split(' ')[0].split(',')) == len(authors_list):
        if len(department_info.split(',')) == len(authors_list):
            author_department_dict_update = dict(zip(authors_list, department_info.split(',')))
        elif len(department_info.split(' ')[0].split(',')) == len(department_info.split(' ')[1].split(',')):
            author_department_dict_update = dict(zip(authors_list, department_info.split(' ')[0].split(',')))
        elif len(department_info.split(' ')[0].split(',')) == len(department_info.split(' ')[1:]):
            author_department_dict_update = dict(zip(authors_list, department_info.split(' ')[0].split(',')))
        else:
            author_department_dict_update = dict(zip(authors_list, department_info.split(' ')[0].split(',')))

    # Case 3: Special case with ' ,'
    elif ' ,' in department_info and len(department_info.split(' ,')) == len(authors_list):
        author_department_dict_update = dict(zip(authors_with_index.split(','), department_info.split(' ,')))

    # Case 4: Departments separated by "!"
    elif department_info.count('!') == len(authors_list) == len(authors_with_index.split(',')):
        departments = re.findall(r'([^,!]+)!', department_info)
        author_department_dict_update = dict(zip(authors_list, departments))

    # Case 5: Fallback to simple splitting
    elif len(department_info.split(',')) == len(authors_list) == len(authors_with_index.split(',')):
        author_department_dict_update = dict(zip(authors_list, department_info.split(',')))

    # Case 6: If none matches, assign all authors the same department
    else:
        author_department_dict_update = {author: department_info for author in authors_list}
        if len(department_info.split(',')) > 2:
            notmatch.add(pid)

    return author_department_dict_update


def process_authorship(input_file, output_file, handprocess_file):
    """
    Process authorship data: split authors, match departments, and clean affiliations.

    :param input_file: path to input CSV
    :param output_file: path to save processed authorship CSV
    :param handprocess_file: pickle file with manually corrected departments
    """
    notmatch = set()
    pattern_city_zip = r'[!\s]?[\u4e00-\u9fa5]+[\s,]?\d{6}'

    # Load manually corrected department mappings
    with open(handprocess_file, 'rb') as f:
        department_HandProcess = pickle.load(f)

    with open(input_file, 'r', encoding='utf-8') as f:
        with open(output_file, 'w', encoding='utf-8', newline='') as fw:
            csv_write = csv.writer(fw)
            csv_head = ["paper id", "title", "author_name", "department", "coauthors", "publish_time", "abstract", "keywords", "journal"]
            csv_write.writerow(csv_head)

            for i, row in enumerate(csv.reader(f)):
                if i == 0:  # Skip header
                    continue

                # Extract author names
                Valid_Chinese_names = [name.replace(" ", "") for name in row[9].split(";")]
                authors_Fulllist = row[2].replace("；", ";").replace(",", ";").replace("，", ";").replace(" ", "").split(";")

                # Case: single author
                if len(authors_Fulllist) == 1:
                    author_name = clean_name_text(authors_Fulllist[0])
                    if row[6] in department_HandProcess:
                        row[6] = department_HandProcess[row[6]]

                    if author_name in Valid_Chinese_names:
                        row[6] = re.sub(pattern_city_zip, '', row[6])
                        if len(author_name) > 1:
                            csv_write.writerow([row[0], row[1], author_name, [row[6]], '', row[4], row[8], row[7], row[3]])

                # Case: multiple authors
                if len(authors_Fulllist) > 1:
                    if row[6] in department_HandProcess:
                        row[6] = department_HandProcess[row[6]]
                    if row[5] in department_HandProcess:
                        row[5] = department_HandProcess[row[5]]

                    author_department_dict_update = diss_depart(row[0], authors_Fulllist, row[5], row[6], notmatch)

                    for j in Valid_Chinese_names:
                        authors_Fulllist, author_department_dict_update = clean_list(authors_Fulllist, author_department_dict_update)
                        authors_Fulllist.remove(j)
                        coauthors = [author for author in authors_Fulllist]

                        if isinstance(author_department_dict_update[j], list):
                            author_department_dict_update[j] = [re.sub(pattern_city_zip, '', text) for text in author_department_dict_update[j]]
                            if len(j) > 1:
                                csv_write.writerow([row[0], row[1], j, author_department_dict_update[j], coauthors, row[4], row[8], row[7], row[3]])
                        else:
                            author_department_dict_update[j] = re.sub(pattern_city_zip, '', author_department_dict_update[j])
                            if len(j) > 1:
                                csv_write.writerow([row[0], row[1], j, [author_department_dict_update[j]], coauthors, row[4], row[8], row[7], row[3]])

                        authors_Fulllist.append(j)


