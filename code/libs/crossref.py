import requests

# Function to get metadata from CrossRef
def get_metadata(title):
    url = f"https://api.crossref.org/works/?query.title={title}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['message']
    else:
        return None

# Function to get citations from OpenCitations
def get_citations(doi):
    url = f"https://opencitations.net/index/api/v1/citations/{doi}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return []

# Function to get references from OpenCitations
def get_references(doi):
    url = f"https://opencitations.net/index/api/v1/references/{doi}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return []

# Function to get metadata for a list of DOIs
def get_metadata_for_list(dois):
    return [get_metadata(doi) for doi in dois]

# Main function to get all required information
def get_paper_data(title):
    data = {}
    metadata = get_metadata(title)
    if metadata:
        data['metadata'] = metadata
        citations = get_citations(doi)
        data['citations'] = [get_metadata(citation['citing']) for citation in citations]
        references = get_references(doi)
        data['references'] = [get_metadata(reference['cited']) for reference in references]
    return data
