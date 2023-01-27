import xml.etree.ElementTree as ET

import logging
logger = logging.getLogger(__name__)
from utils.relevant_tags import tag_locations


def get_root(file_location):
    """
    Open the file location and parse it to xml.
    Args:
        file_location (str): The location of the xml file.
    Returns:
        tree (xml.etree.ElementTree): The parsed xml tree.
    """
    try:
        f = open(file_location, 'r')
        tree = ET.parse(f)
        return tree
    except ET.ParseError:
        return


def get_article_type(tree, article_type):
    """
    Check if the article type is matching the desired type.
    Args:
        tree (xml.etree.ElementTree): The parsed xml tree.
        article_type (str): The desired article type.
    Returns:
        xml.etree.Element: The article element if the article type matches the desired type,
        None otherwise.
    """
    article = tree.find('.//article')
    if article.attrib['article-type'] == article_type:
        return article


def extract_content(root, level, tag=None, attr=None, attr_val=None):
    """
    Extract the content from the xml.
    Args:
        root (xml.etree.ElementTree): The parsed xml tree.
        level (str): The level to extract the content from.
        tag (str): The tag to extract the content from.
        attr (str): The attribute of the tag to extract the content from.
        attr_val (str or list of str): The attribute value of the tag to extract the content from.
    Returns:
        str: the content extracted from the xml.
    """

    result = set()
    root_level = root.find(f'.//{level}/')
    if tag:
        for element in root_level:
        #for element in root.iterall(f'.//{level}/'):

            if element.tag == tag:
                print(element.tag)

                for el in element.iter():
                    if el.attrib.get(attr, '').strip().lower() in attr_val:
                        text = ''.join(el.itertext())
                        if text:
                            text = '\n'.join(text.split())
                            result.add(text)
            # else:
            #     for sub_element in root.findall(f'.//{level}/{element.tag}/'):
            #         # print(sub_element.tag)
            #         if sub_element.tag == tag:
            #             # print(sub_element.tag, sub_element.attrib)
            #             print(attr_val)
            #             if sub_element.attrib.get(attr, '').strip().lower() in attr_val:
            #                 text = ''.join(sub_element.itertext())
            #                 print(sub_element.attrib)
            #                 if text:
            #                     text = '\n'.join(text.split())
            #                     result.add(text)

    # print(root, len(result))

    if len(result) > 0:
        return '\n'.join(result)




def return_unique_dict(values):

    # Workout to avoid doing several time the same parsing
    # Should be put outside loop but require to rewrite everything

    already_done = set()
    for value in values:
        val_to_access = tag_locations[value]
        level = val_to_access['level']
        tag = val_to_access['tag']
        attr = val_to_access['attr']
        attr_val = val_to_access['attr_val']
        final_val = list()

        if isinstance(attr_val, str):
            attr_val = [attr_val]

        for val in attr_val:
            to_check = f'{level}_{tag}_{attr}_{val}'
            if to_check not in already_done:
                final_val.append(val)
                already_done.add(to_check)
        
        yield level, tag, attr, final_val
        


def extract_value(xml, values):
    """
    Extract the desired values from the xml.
    Args:
        xml (xml.etree.ElementTree): The parsed xml tree.
        values (str or list of str): The values to extract from the xml.
    Returns:
        str: the content extracted from the xml.
    """
    # As they can have duplicate results when passing different values
    # Use set to remove them
    # Todo: rather than set, should avoid to parse several time the same
    # Section and check the values from the relevant_tags dictionary before
    
    results = set()
    if isinstance(values, str):
        values = [values]
    for level, tag, attr, attr_val in return_unique_dict(values):
        attr_val = "".join(attr_val)
        result = extract_content(xml, level, tag, attr, attr_val)
        if result:
            results.add(result)
    try:
        results = '\n'.join(results)
        return results
    except TypeError:
        return


def get_content(file_location, values, article_type='research-article'):
    """
    Get the content from the xml file.
    Args:
        file_location (str): The location of the xml file.
        values (str or list of str): The values to extract from the xml.
        article_type (str): The desired article type.
    Returns:
        str: the content extracted from the xml.
    """
    xml = get_root(file_location)
    if xml:
        root = get_article_type(xml, article_type)
        if root:
            return extract_value(root, values)


def multiple_content(file_location, dictionary_values, article_type='research-article'):
    """
    Get multiple content from the xml file.
    Args:
        file_location (str): The location of the xml file.
        dictionary_values (dict): The different values needed to be return with each key a separated one and
        the value where to find it
        article_type (str): The desired article type.
    Returns:
        dict: the content extracted from the xml per key
    """
    dict_result = dict()
    for content in dictionary_values:
        result = get_content(file_location, dictionary_values[content], article_type)
        dict_result[content] = result
    return dict_result
