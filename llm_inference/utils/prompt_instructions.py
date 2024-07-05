import json


json_response_format = {
    "answer": "str",
    "labels": {
        "sample": {
            "total": "int",
            "sample": "[int]",
            "sentence_where_found": "[str]",
        },
        "male": {
            "total": "int",
            "sample": "[int]",
            "sentence_where_found": "[str]",
        },
        "female": {
            "total": "int",
            "sample": "[int]",
            "sentence_where_found": "[str]",
        },
    },
}

# json_response_format = {
#     "answer": "str",
#     "labels": {
#         "n_female": "int",
#         "n_female_p": "[int]",
#         "n_male": "int",
#         "n_male_p": "[int]",
#         "perc_female": "int",
#         "perc_male": "int",
#         "perc_female_p": "[int]",
#         "perc_male_p": "[int]",
#     },
# }


json_string = json.dumps(json_response_format, indent=4)

prompt_instruction_1 = f"""Find the total of men and women in that study. 
Your response should be in JSON only. If you need to output information about your process do it under the key: model_inference_info and don't output a Note at the end. 
If there are several sample that contains men or women, create a list of sample size for men and a list of sample size for women in addition to the total sample size per women and men and the total. 
Women can be sometimes refered to female. Men can be sometimes refers to males. 
In case there are percentages. output a field specifying it is percentages. 
If there is no mention of men or women (or related terms) please state that the value are 0, don't assume the representation. 
For the JSON verify it follow this structure: {json_string}"""

prompt_instruction_2 = f"""data_source": "Text extracted from research article",
"processing_steps": [
"Identified key sections discussing men and women in the study",
"Extracted relevant numerical data on men and women",
"Validated the data against the context in the text",
"Calculated total sample size, men and women counts",
"Don't make assumption of the repartition if no direct mentions",
“If the answer is clear, stated in answer: `accept`”,
“If it is not about human subject stated in answer: `ignore`”,
“If the information is not clear stated in answer: `reject`”,
"Output the decision process  in json format following the format: {json_string}"""


prompt_instruction_3 = f"""
data_source": "Text extracted from research article",
"processing_steps": [
"Identified key sections discussing men and women in the study",
"Extracted relevant numerical data on men and women",
"Validated the data against the context in the text",
"Calculated total sample size, men and women counts",]

To take a decision follow these instructions: 

Only interested in the number of human subjects: E.g.:  individuals, participants, males, females, lung from 40 patients, Human subjects (single and plural)
We exclude: Cells, strains or cells lines. E.g. 30 cells, what about human parts. Animals different from human. E.g. 40 mice, 
One subject may have multiple samples. We want the number of individuals not the number of samples. E.g. 20 samples from 10 patients. (N=10).
Use partial numbers when you are sure that they can be added.
Filtering samples. Options:
We will take the largest number of samples.
There can be multiple filtering steps, different subsets of samples used for different experiments,..
When the sex of the samples is not explicitly there but it is clear from the context, annotate the samples as N_males / N_females instead of N_samples. A number correspond to two labels? E.g. Study about pregnancy: 10 subjects… (N_females = 10 & N=10) → Accept

After annotating, select:
Accept: All information is clear and about human subject 
Reject: There are label(s) but at least one is confusing.
There is at least 1 label that you are doubting if or how to annotate based on the provided rules.
Ignore: There are no labels of interest. For instance, the paper is about animals (not human), cell lines, strains, etc. In other words, there are no numbers or percentages of human subjects, males or females. 

Only answer the json no added text.
If there is no text, do not invent response. Only output the json with Ignore
The answer should be in json format:\n{json_string}
The text you need to extract the information from:
"""
