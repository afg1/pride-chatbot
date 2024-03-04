"""
This script search for all the datasets in PRIDE with SDRF files and create an md file for each dataset with
the given metadata information:
- Dataset accession
- Title
- Description
- Species
- Tissue
- Disease
- Instrument
- Sample Protocol
- Data Protocol
- Publication Abstract
- Publication Title
- Number of raw files
- For all the SDRs in the dataset get:
- characterists with all the given values.
"""

import pandas as pd
import requests
import json
import argparse
import os
import time
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def search_all_projects_with_sdrf(size=100, page=0):
    """
    This function search for all the datasets in PRIDE with SDRF files and return a list with all the project accessions
    """
    url = 'https://www.ebi.ac.uk/pride/ws/archive/v2/search/projects?sortDirection=DESC&page={}&pageSize={}&dateGap=%2B1YEAR'.format(
        page, size)
    response = requests.get(url)
    data = json.loads(response.text)

    project_accessions = []
    if '_embedded' in data and 'compactprojects' in data['_embedded']:
        for project in data['_embedded']['compactprojects']:
            project_accessions.append(project)

    return project_accessions


### Main function
def get_abstract_from_doi(doi: str):
    base_url = "https://api.crossref.org/works/"
    search_url = f"{base_url}{doi}"

    try:
        response = requests.get(search_url)
        data = response.json()
        abstract = data['message']['abstract']
        return abstract
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_abstract_from_pubmed(pubmed_id:int):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id="
    search_url = f"{base_url}{pubmed_id}&retmode=xml"

    try:
        response = requests.get(search_url)
        data = response.text
        abstract = re.search(r'<AbstractText>(.*?)</AbstractText>', data).group(1)
        return abstract
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_pubmed_id_from_doi(doi):
    """Get the pubmedi from doi using europepmc"""
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
    search_url = f"{base_url}{doi}&format=json"

    try:
        response = requests.get(search_url)
        data = response.json()
        pubmed_id = data['resultList']['result'][0]['pmid']
        return pubmed_id
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_project_details(accession):
    url = f'https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{accession}'
    print(url)
    response = requests.get(url)
    data = json.loads(response.text)

    # url = f'https://www.ebi.ac.uk/pride/ws/archive/v2/files/getCountOfFilesByType?accession={accession}'
    # print(url)
    # response = requests.get(url)
    # file_data = json.loads(response.text)
    # data['number_raw_files'] = file_data['RAW']

    url = f'https://www.ebi.ac.uk/pride/ws/archive/v2/files/sdrfByProjectAccession?accession={accession}'
    print(url)
    try:
        sdrf_df = pd.read_csv(url, sep='\t')
        data['sdrf'] = sdrf_df
    except:
        data['sdrf'] = None
    abstract = None
    if 'references' in data:
        references = data['references']
        for reference in references:
            if reference['pubmedId'] == 0 and len(reference['doi']) > 0:
                pubmed_id = get_pubmed_id_from_doi(reference['doi'])
                abstract = get_abstract_from_pubmed(pubmed_id)
            elif reference['pubmedId'] > 0:
                abstract = get_abstract_from_pubmed(reference['pubmedId'])
    data['publication_abstract'] = abstract
    return data


def remove_emails_from_contacts(project):
    if project["accession"] == 'PXD000759':
        logger.info(f"Removing emails from project {project['accession']}")
    if 'labPIs' in project:
        for labIds in project['labPIs']:
            if 'email' in labIds:
                labIds['email'] = ''
    return project


def get_string_labpi(lab_pi):
    string = ""
    for labpi in lab_pi:
        if 'title' in labpi:
            string += labpi['title']
        if 'name' in labpi:
            string += " " + labpi['name']
        if 'affiliation' in labpi:
            string += "\n" + labpi['affiliation']
        string += "\n"
    return string


def proccess_values_in_sdrf(unique_values):
    values = []
    for value in unique_values:
        if isinstance(value, str) and ('not applicable' in value or 'not available'):
            print("Not available value")
        if isinstance(value, str) and "NT=" in value:
            sub_values = value.split(";")
            for value in sub_values:
                if "NT=" in value:
                    value = value.replace("NT=", "")
                    values.append(value)
        else:
            values.append(value)
    return values


def main():
    parser = argparse.ArgumentParser(description='Create md files for all the datasets in PRIDE with SDRF files')
    parser.add_argument('--output_folder', dest='output_folder', required=False, help='Output folder for the md files')
    args = parser.parse_args()

    if args.output_folder:
        output_folder = args.output_folder
    else:
        output_folder = 'metadata'

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all the project accessions with SDRF files
    logger.info('Get all the project accessions with SDRF files')
    # Get projects recursively
    project_accessions = []
    page = 0
    while True:
        logger.info(f'Getting page {page}')
        projects = search_all_projects_with_sdrf(page=page)
        if len(projects) == 0:
            break
        project_accessions.extend(projects)
        page += 1

    print("The number of projects is: ", len(project_accessions))

    keywords_to_search = ['disease', 'organism', 'organism part', 'instrument', 'sample type', 'tissue', 'cell type', 'strain/breed', 'separation', 'fractionation method]', 'label', 'modification parameters']
    for project in project_accessions:
        accession = project['accession']
        title = project['title']
        description = project['projectDescription']
        sample_protocol = project['sampleProcessingProtocol']
        data_protocol = project['dataProcessingProtocol']
        project_detail = get_project_details(accession)
        lab_pi = project_detail['labPIs'] if 'labPIs' in project_detail else ''
        print(project_detail)
        project_detail = remove_emails_from_contacts(project_detail)
        with open(f'{output_folder}/{accession}.md', 'w') as f:
            f.write('### Accession\n{}\n\n'.format(accession))
            f.write('### Title\n{}\n\n'.format(title))
            f.write('### Description\n{}\n\n'.format(description))
            f.write('### Sample Protocol\n{}\n\n'.format(sample_protocol))
            f.write('### Data Protocol\n{}\n\n'.format(data_protocol))
            f.write('### Publication Abstract\n{}\n\n'.format(project_detail['publication_abstract']))
            f.write('### Keywords\n{}\n\n'.format(', '.join(project['keywords'])))
            f.write('### Affiliations\n{}\n\n'.format('\n'.join(project['affiliations'])))
            f.write('### Submitter\n{}\n\n'.format('\n'.join(project['submitters'])))
            f.write('### Lab Head\n{}\n\n'.format(get_string_labpi(lab_pi)))
            if project_detail['sdrf'] is not None:
                f.write('### SDRF\n')
                for column in project_detail['sdrf'].columns:
                    if any(keyword in column.lower() for keyword in keywords_to_search):
                        unique_values = project_detail['sdrf'][column].unique()
                        column = column.replace(']', '').replace('characteristics[', '').replace('comment[', '').replace("Characteristics[", "")
                        pattern = r'\.\d+'
                        column = re.sub(pattern, '', column)
                        f.write(f"- {column}: {', '.join(str(value) for value in proccess_values_in_sdrf(unique_values))}\n")
                f.write('\n')

if __name__ == '__main__':
    main()
