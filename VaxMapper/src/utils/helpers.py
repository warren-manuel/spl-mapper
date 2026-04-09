# Contents of /rxnorm-term-getter/rxnorm-term-getter/src/utils/helpers.py

import pandas as pd
import requests
from owlready2 import get_ontology, World

def get_rxnorm_data(rxcui):
    result = {}
    result['rxcui'] = rxcui
    status_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/historystatus.json?"
    try:
        response = requests.get(status_url)
        if response.status_code == 200:
            data = response.json()
            result['name'] = data.get("rxcuiStatusHistory", {}).get("attributes", {}).get("name")
            result['status'] = data.get("rxcuiStatusHistory", {}).get("metaData", {}).get("status")
            result['tty'] = data.get("rxcuiStatusHistory", {}).get("attributes", {}).get("tty")
    except requests.RequestException as e:
        print(f"Error fetching RxNorm status for {rxcui}: {e}")
    return result

def get_class_descendants(class_id, class_type):
    api_url = f"https://rxnav.nlm.nih.gov/REST/rxclass/classTree.json?classId={class_id}&classType={class_type}"
    descendants = []
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()      
        data = response.json()
        descendants = _collect_leaf_classids(data["rxclassTree"])
        
    except Exception as e:
        print(f"API error with {class_id}: {e}")
        
    return descendants

def _collect_leaf_classids(tree):
    leaves = []
    for node in tree:
        cid = node["rxclassMinConceptItem"]["classId"]
        cname = node["rxclassMinConceptItem"]["className"]
        children = node.get("rxclassTree")
        if children:                      
            leaves.extend(_collect_leaf_classids(children))
        else:                             
            leaves.append({'ID': cid, 'Name': cname})
    return leaves

def get_rx_class_members(params):
    api_url = f"https://rxnav.nlm.nih.gov/REST/rxclass/classMembers.json"
    class_id = params.get("classId")
    source = params.get("relaSource")
    result = {"class_id": class_id, "source": source, "related_ids": None}
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()      
        data = response.json()
        if "drugMemberGroup" in data and "drugMember" in data["drugMemberGroup"]:
            related_ids = [item["minConcept"]["rxcui"] for item in data["drugMemberGroup"]["drugMember"]]
            result["related_ids"] = related_ids
    except Exception as e:
        print(f"API error with {class_id}: {e}")

    return result

def get_related_rxcui(rxcui):
    api_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allrelated.json"
    result = {"rxcui": rxcui, "related_ids": None}

    try:
        response = requests.get(api_url)
        response.raise_for_status()      
        data = response.json()
        concept_groups = data.get("allRelatedGroup", {}).get("conceptGroup", [])
        related_concepts = []
        for group in concept_groups:
            if "conceptProperties" in group:
                related_concepts.extend([concept.get("rxcui") for concept in group["conceptProperties"]])

        result["related_ids"] = related_concepts

    except Exception as e:
        print(f"API error with {rxcui}: {e}")

    return result