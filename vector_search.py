from collections import defaultdict

import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


def vector_by_id(path_id: str):
    directory = "./vector/" + path_id
    vector = Chroma(persist_directory=directory,
                    embedding_function=HuggingFaceEmbeddings(model_name='paraphrase-MiniLM-L6-v2'))
    data = vector.get()['metadatas']
    unique_data = []
    seen = set()

    for item in data:
        identifier = item['source']
        if identifier not in seen:
            seen.add(identifier)
            unique_data.append(item)
    vector.source = unique_data
    return vector


def get_similar_projects_from_solr(accessions_str):
    accession_list = accessions_str.split(',')
    url = "https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{accession}/similarProjects?pageSize=10"
    accession_title_counts = defaultdict(int)

    for accession in accession_list:
        response = requests.get(url.format(accession=accession))
        if response.status_code == 200:
            data = response.json()
            compact_projects = data["_embedded"]["compactprojects"]
            for project in compact_projects:
                accession_title_counts[(project["accession"], project["title"])] += 1
                print(project["accession"] + " " + project["title"] + " " +
                      str(accession_title_counts[(project["accession"], project["title"])]))
        else:
            print(f"Failed to fetch data for accession {accession}")

    accession_title_list = [{"accession": accession, "title": title, "count": count} for (accession, title), count in
                            accession_title_counts.items()]
    sorted_accession_title_list = sorted(accession_title_list, key=lambda x: x["count"], reverse=True)
    return sorted_accession_title_list


if __name__ == '__main__':
    # start service

    ve = vector_by_id('d4a1cccb-a9ae-43d1-8f1f-9919c90ad370')
    a = 'Are there any metabolomics data in PRIDE?'
    for d in ve.similarity_search_with_score(a, k=3):
        print(d)
        print('\n----------------------------------------------\n')
