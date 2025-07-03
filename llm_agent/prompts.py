from dataclasses import dataclass

from typing import Dict, Any


end_instruction = """ 
    Strictly respect the following instructions:
    1. If the document is empty, just answer False; 
    2. If there is enough information to answer the question, answer True or False, but do NOT try to guess it
    3. Every sentences in the text, tables, exerpts need to be collected and added to the context field as a list. Be exhaustive.
    4. If there is a table (in string format), do extract it in one string, not in multiple strings.
    5. Follow the format instructions strictly and do not provide any additional information beyond what is requested
    """




@dataclass
class PromptTemplate:
    """Generic dataclass for prompt templates."""
    system_prompt: str
    instruction_prompt: str = "" 




def human_presence(section_header: str, section_content: str, end_instruction: str) -> PromptTemplate:
    """Generates the prompt to know if the article is about human subjects"""
    system_prompt = f"""You are a helpful assistant analyzing a scientific article. You can call tools to extract specific sections of the document."""

    instruction_prompt = f"""
    Your task is to determine if the article involves a study on human subjects. 
    Focus on the following section: {section_header} of the document: {section_content}
    
    Respond with:
    - 'True': If the document explicitly mentions human subjects.
    - 'False': If the document does not mention human subjects or lacks enough information for a clear decision.
    
    {end_instruction}
    """
    return PromptTemplate(system_prompt=system_prompt, instruction_prompt=instruction_prompt)