# Three tags are present in all documents (almost), Front, sec and back.
# They contains different type of information about the articles
# Parsing them differently can give more complete information
# These keywords are single and lowercase,
# Extracted from the sec level if they are > 5 occurences among the entire dataset dl with human subject as filter
# Ordered by descending count
# They are found using the notebook ./tag_exploration.ipynb

# The information for front is in the embedded in the tags
value_sections = {
    "INTRO": [
        "intro",
        "introduction",
        "background",
        "intro|methods",
        "intro|subjects",
        "general",
        "summary",
        "intro|methods|subjects",
        "objectives",
        "opening-section",
        "aim",
        "intro|results",
        "opening",
        "description",
    ],
    "METHODS": [
        "materials|methods",
        "methods",
        "subjects",
        "materials and methods",
        "materials-and-methods",
        "materials",
        "methods|subjects",
        "material|methods",
        "subjects|methods",
        "materials-methods",
        "intro|methods",
        "intro|subjects",
        "materials | methods",
        "methods|results",
        "methods|materials",
        "method",
        "patients|methods",
        "materialsandmethods",
        "material and methods",
        "materials methods",
        "intro|methods|subjects",
        "methods|discussion",
        "materials and method",
        "materials|method",
        "materialandmethods",
        "material|method",
        "supplemental|material",
        "patients and methods",
        "survey|methodology",
        "methods|conclusions",
        "materials|methods",
        "materials|methodology",
        "methods|cases",
        "methods and materials",
        "methodology",
        "intro|methods|results",
        "material|methods",
    ],
    "SUBJECTS": [
        "subjects",
        "methods|subjects",
        "subjects|methods",
        "intro|subjects",
        "patients|methods",
        "intro|methods|subjects",
        "patients and methods",
        "subjects|results",
        "subjects|discussion",
        "sampling description",
        "species",
    ],
    "RESULTS": [
        "results",
        "results|discussion",
        "result",
        "methods|results",
        "results and discussion",
        "results|conclusions",
        "results | discussion",
        "discussion|results",
        "results|discussion",
        "intro|results",
        "intro|methods|results",
        "finding",
        "diagnosis",
    ],
    "DISCUSSION": [
        "discussion",
        "discussions",
        "results|discussions",
        "discussions",
        "results|discussion",
        "disucssion|conclusions",
        "results and discussion",
        "discussion|conclusion",
        "discussion|interpretation",
        "methods|discussion",
        "results | discussion",
        "discuss",
        "discusion",
        "discussion|results",
        "subjects|discussion",
        "result|discussion",
        "discussion | conclusions",
        "conclusion|discussion",
    ],
    "CONCLUSION": [
        "conclusions",
        "conclusion",
        "discussion|conclusions",
        "summary",
        "discussion|conclusion",
        "perspectives",
        "results|conclusions",
        "limitations",
        "discussion | conclusions",
        "conclusion|discussion",
        "research highlights",
        "key messages",
    ],
    "ACK_FUND": [
        "funding-information",
        "acknowledgement",
        "acknowledgment",
        "funding",
        "acknowledgements",
        "financial-disclosure",
        "funding sources",
        "funding-statement",
        "aknowledgement",
        "acknowlegement",
        "open access",
        "financial support",
        "grant",
        "author note",
        "financial disclosure",
    ],
}
