# Instruction to get the annotations and train a model on them

[!warning] The python scripts files have been moved from their original folder and the paths to data need to be changed accordingly

## Setting up Prodigy for getting annotation



In a VM with prodigy installed: 

```
python csv_to_jsonl.py

source prodigyenv/bin/activate

prodigy spans.manual biohack blank:ca raw.jsonl -l n_fem,n_male,perc_fem,perc_male,sample

prodigy db-out biohack > data/annotations.jsonl

```


## Transform the annotations to conll format


Convert to conll and create the splits for training:

```
python training_data/annotations_to_conll.py --data .annotation/annotations.jsonl
```

## Train the model to extract the number of male and female in the server

```
sh train_model.sh
```

## Use the model to infere where is the info

```
cd ..
python get_sex_bias.py --data data/candidate_sentences_last.csv --out results.json --model train_model/model
```
