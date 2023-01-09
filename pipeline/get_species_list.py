import requests
from requests.exceptions import HTTPError
import concurrent.futures
import random
import logging
import os
import yaml
from tqdm import tqdm
import urllib.parse

logger = logging.getLogger(__name__)


config_path = os.path.join(os.path.dirname(
    __file__), '../config', 'config.yaml')
config_all = yaml.safe_load(open(config_path))


def get_request(url, payload):
    params = urllib.parse.urlencode(payload, quote_via=urllib.parse.quote)
    try:
        response = requests.get(url, params)
        # If the response was successful, no Exception will be raised
        response.raise_for_status()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')
    else:
        return response


def get_species(pmcid, annotation_url):
    url = annotation_url
    pmcid_to_load = f'PMC:{pmcid[3:]}'
    payload = {'articleIds': pmcid_to_load,
               'type': 'Organisms',
               'section': 'Methods',
               'provider': 'Europe PMC',
               # 'provider': requests.utils.quote('Europe PMC'),
               'format': 'JSON'}
    result = get_request(url, payload)
    try:

        result_json = result.json()
        species = None
        for dictionary in result_json:
            for d in dictionary['annotations']:
                species = d['exact']
                break
        
        print(pmcid, species)
        return pmcid, species

    except AttributeError:  # When not getting anything
        pass


def get_pmcidlist(file_location):
    with open(file_location, 'r') as f:
        for l in f:
            yield l.rstrip()


def get_parsed_list_species(file_location):
    try:
        with open(file_location, 'r') as f:
            for l in f:
                yield l.split(',')[0].rstrip()
    except FileNotFoundError:
        pass


def get_list_to_dl(pmcid_archive_location, pmcid_species_location):

    pmcid_to_dl = set(get_pmcidlist(pmcid_archive_location))
    print(f"Len of pmcid_to_dl: {len(pmcid_to_dl)}")

    list_parsed_species = set(
        get_parsed_list_species(pmcid_species_location))
    print(f"Len of already done species: {len(list_parsed_species)}")

    ids_to_dl = list(pmcid_to_dl.difference(list_parsed_species))
    print(f"Len of pmcid to parse: {len(ids_to_dl)}")
    # Randomize the list to avoid downloading only the first articles
    random.shuffle(ids_to_dl)
    return ids_to_dl


def main():
    annotation_api = config_all['api_europepmc_params']['annotations_api']['root_url']

    pmcid_archive_location = config_all['api_europepmc_params']['pmcid_archive_location']
    pmcid_species_location = config_all['api_europepmc_params']['pmcid_species_location']

    ids_to_dl = get_list_to_dl(pmcid_archive_location, pmcid_species_location)
    futures = []
    executor = concurrent.futures.ThreadPoolExecutor()
    print('Starting the process')
    species_dictionary = dict()
    futures = [executor.submit(get_species, pmcid, annotation_api)
               for pmcid in ids_to_dl]

    print('Process started. Getting results')
    pbar = tqdm(total=len(ids_to_dl))  # Init pbar
    with open(pmcid_species_location, 'a') as f:
        for future in concurrent.futures.as_completed(futures):

            result_pmcid, species = future.result()
            try:

                species_dictionary[species] += 1
            except KeyError:
                species_dictionary[species] = 1

            f.write(f"{result_pmcid},{species}")
            f.write('\n')
            pbar.update(n=1)
            exception = future.exception()
            if exception:
                raise(exception)


if __name__ == "__main__":
    main()
