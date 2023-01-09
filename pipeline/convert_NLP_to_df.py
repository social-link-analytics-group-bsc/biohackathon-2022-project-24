
import pandas as pd
import numpy as np


model_df = pd.read_json("./data/results_nlp.json").T
model_df_joined = model_df.copy(deep=True)


model_df_joined.female = model_df_joined.female.apply(lambda y: y[0] if len(y)==1 and isinstance(y, list) else y)
model_df_joined.male = model_df_joined.male.apply(lambda y: y[0] if len(y)==1 and isinstance(y, list) else y)

model_df_joined.female = model_df_joined.female.apply(lambda y: np.nan if len(y)==0 and isinstance(y, list) else y)
model_df_joined.male = model_df_joined.male.apply(lambda y: np.nan if len(y)==0 and isinstance(y, list) else y)

model_df_joined.female_percentage = model_df_joined.female_percentage.apply(lambda y: np.nan if len(y)==0 else y)
model_df_joined.male_percentage = model_df_joined.male_percentage.apply(lambda y: np.nan if len(y)==0 else y)


def hashtag_modifier(dataframe, column_name):
    for (ix, row), i in zip(dataframe.iterrows(), dataframe[column_name]):

        if isinstance(i, list) and "##" in ("").join(i):
            # print(i, "-- this row will be modified")
            joined_nums = model_df_joined[column_name][ix]
            relevant = [item for item in joined_nums]
            relevant_id = [ix for ix, elem in enumerate(model_df_joined[column_name][ix]) if "##" in elem]

            indexes = []
            new_relevant = []
            final_relevant = []

            for item, id in zip(relevant, relevant_id):
                tmp_list = relevant
                tmp_list_joint = []
                if  int(id+1) <= len(tmp_list)-1 and "##" in tmp_list[int(id+1)]:
                    tmp_list_joint.append(''.join(tmp_list[int(id-1) : int(id+2)]))
                
                if  int(id+1) <= len(tmp_list)-1 and "##" not in tmp_list[int(id+1)]:
                    tmp_list_joint.append(''.join(tmp_list[int(id-1) : int(id+1)]))
                
                if int(id) == len(tmp_list)-1 and "##" in tmp_list[int(id)] and "##" not in tmp_list[int(id-1)]:
                    tmp_list_joint.append(''.join(tmp_list[int(id-1):]))

                new_number = str(tmp_list[0])
                new_number.replace("##", "")
                new_relevant.append(tmp_list_joint)

            for ix, item in enumerate(relevant):
                try:
                    if "##" not in relevant[ix+1] and "##" not in relevant[ix]  and ix <= len(relevant)-1:
                        final_relevant.append(item)

                except:
                    pass

                if ix == len(relevant)-1 and "##" not in relevant[ix]:
                    final_relevant.append(item)
            
            for i in new_relevant:
                if len(i) > 0:
                    final_relevant.append(str(i[0]).replace("##", ""))
            
            final_relevant = [int(float(x)) for x in final_relevant if len(x) != 0] # avoid empty strings

            if "#" in ("").join(joined_nums):
                # print(ix, "this is the new row", final_relevant, "with max value: ", max(final_relevant))
                row[column_name] =  max(final_relevant)
                # print("the row to be modified is", row["male"])

        elif not isinstance(i, list) and "##" not in str(i) and "~" not in str(i):
            # print(i, "CASE 2")
            pass
        elif isinstance(i, list) and "##" not in ("").join(i): # there's an empty string that throws a ValueError
            # print(ix, "with max value", max([int(float(x)) for x in i if len(x) != 0]))
            row[column_name] = max([int(float(x)) for x in i if len(x) != 0])
            pass


hashtag_modifier(model_df_joined, "male")
hashtag_modifier(model_df_joined, "female")
hashtag_modifier(model_df_joined, "female_percentage")
hashtag_modifier(model_df_joined, "male_percentage")

model_df_joined.reset_index(inplace=True)
model_df_joined.rename({"index":"PMCID"}, axis=1, inplace=True)
model_df_joined.to_csv("./data/NLP_counts.csv", sep=',', encoding='utf-8', index=False)


# join with provided df

sentences_df = pd.read_csv("./data/CS_v3.csv")

final_df_small = pd.merge(sentences_df, model_df_joined, on=["PMCID"])
final_df = pd.merge(sentences_df, model_df_joined, how="outer", on=["PMCID"])

final_df.drop(["Unnamed: 0"], axis=1, inplace=True) # if needed, only
final_df.to_csv("./data/CS_def.csv", sep=',', encoding='utf-8', index=False)
