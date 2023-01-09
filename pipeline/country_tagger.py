import pandas as pd
import spacy
import os
from math import isnan
from tqdm import tqdm


# if you get an error you need to downoald
# python3 -m spacy download en_core_web_sm
# 


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")


    if not os.path.isfile("CS_v0.csv"):
        df = pd.read_csv("./data/CS_v0.csv")
        
        for idx in tqdm(df.itertuples()):
            if isinstance(idx.aff,str):
                doc = nlp(idx.aff)
                for e in doc.ents:
                    if e.label_=="GPE":  
                        df.loc[idx.Index, 'Countries'] = e.text
                    if e.label_=="ORG":  
                        df.loc[idx.Index, 'Organisations'] = e.text
            
        # df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1'], axis=1, inplace=True)
        df.to_csv("CS_v1.csv", index=False)
    else:
        print("file exists...")