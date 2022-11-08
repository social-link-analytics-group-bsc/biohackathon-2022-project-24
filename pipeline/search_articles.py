import requests
from requests.exceptions import HTTPError

import xml.etree.ElementTree as ET
import gzip
import io
import logging
import os
import yaml
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)


config_path = os.path.join(os.path.dirname(
    __file__), '../config', 'config.yaml')
config_all = yaml.safe_load(open(config_path))


def get_request(url, params):
    try:
        response = requests.get(url, params)
        # If the response was successful, no Exception will be raised
        response.raise_for_status()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')
    else:
        return requests.get(url, params)


def get_archive(file_location, url, rerun=False):

    if rerun is False:
        try:
            open_f = open(file_location, 'rb')
            f = io.BytesIO(open_f.read())
        except FileNotFoundError:
            rerun = True
    if rerun is True:
        logger.info(f"Getting the archive at {url}")

        OAUrl = requests.get(url)
        gzFile = OAUrl.content
        location = open(file_location, 'wb')
        location.write(gzFile)
        f = io.BytesIO(gzFile)
        logger.info(f"Writing archive in {file_location}")
    with gzip.GzipFile(fileobj=f) as OAFiles:
        n = 0
        for OAFile in OAFiles:
            yield str(OAFile[:-1], 'utf-8')
            n += 1
            if n == 100000:
                return


def getting_pmcids(query, root_url):
    req = f'{root_url}search/query={query}&resultType=idlist&cursorMark=*&pageSize=100'
    r = requests.get(req)
    ids = []
    while True:
        root = ET.fromstring(r.content)
        pmcids = retrievePmcids(root)
        for ids in pmcids:
            yield ids
        nextpage = root.find('nextPageUrl')
        if nextpage is None:
            break
        r = requests.get(nextpage.text)


def retrievePmcids(root):
    pmcids = []
    for e in root.iter('result'):
        if e.find('pmcid') is not None:
            pmcids.append(e.find('pmcid').text)
    return pmcids


def main():
    api_root_article = config_all['api_europepmc_params']['rest_articles']['root_url']
    file_name = config_all['search_params']['ids_file_location']
    query = config_all['search_params']['query']
    with open(file_name, 'w') as f:
        for pmcid in getting_pmcids(query, api_root_article):
            print(pmcid)
            f.write(pmcid)
            f.write('\n')


if __name__ == "__main__":
    main()
