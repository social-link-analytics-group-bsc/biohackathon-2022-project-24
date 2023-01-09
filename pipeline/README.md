# pipeline

## Download EuropePMC samples from list of pmcids

```
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

To download the sample data and transform to json:

```
python use_api.py 
```

Currently the sample comes from the query "eLIfe AND ((HAS_FT:Y AND OPEN_ACCESS:Y)) AND (((SRC:MED OR SRC:PMC OR SRC:AGR OR SRC:CBA) NOT (PUB_TYPE:"Review")))"

If you want to explore different sets of data go to: <https://europepmc.org/>

* Do your query
* Download the ids in "Export citations" and save the result in data/
* Indicate the new file in the config "ids_file_location"

## Explore the sections

### Tables

Explore all the tables in the jsons by running (we did not use it, no info in the tables):

```
python explore_tables.py
```

### Methods

Find the sentences containing the tokens ['man', 'woman', 'male', 'female', 'men', 'women', 'males', 'females'] with a number around it:

```
python explore_methods.py
```

## get article info

get metadata info given a csv containing pmcids and directory containing the articles (xml files)

```
python more_filtering.py -f data/ids-10.csv -d data/articles/        
```

get annotation info given a csv containing pmcids and an entity type (e.g., `Organisms`)

```
python add_annotation_info.py -f data/ids-10.csv  -a Organisms

python add_annotation_info.py -f data/ids-10.csv  -a Diseases
```

List of entities :

```
* Organisms
* Diseases
* Accessions
* Genes/Proteins
* Chemicals
* Gene Ontology
* Resources
* Experimental Methods
* [and more else...](https://europepmc.org/AnnotationsApi#/)
```

get list of mesh terms associated to articles given a csv containing pmcids

```
add_metadata.py -f new_data.csv  -m mesh
```
