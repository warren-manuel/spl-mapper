import pandas as pd
import requests
from owlready2 import get_ontology, World

def identify_missing_rxnorm_terms(vo_path_or_url=None):
    """
    Identifies RxNorm terms related to vaccines that are not present in the current version of VO.
    
    Args:
        vo_path_or_url (str): Path or URL to the VO ontology file. 
                             Defaults to PURL if None.
    
    Returns:
        pd.DataFrame: DataFrame containing information about missing RxNorm terms
    """
    
    print("Step 1: Loading VO and extracting existing RxNorm terms...")
    # 1. Load the VO ontology
    onto_w = World()
    if vo_path_or_url is None:
        vo_path_or_url = "http://purl.obolibrary.org/obo/vo.owl"
    
    onto = onto_w.get_ontology(vo_path_or_url).load()
    
    # 2. Extract existing RxNorm IDs
    RXNORM_PROP_IRI = 'http://purl.obolibrary.org/obo/VO_0003198'
    rxnorm_prop = onto_w[RXNORM_PROP_IRI]
    
    rxnorm_ids = set()
    
    # Extract from classes
    for cls in onto.classes():
        if hasattr(cls, rxnorm_prop.name):
            for rid in getattr(cls, rxnorm_prop.name):
                rxnorm_ids.add(str(rid))
    
    # Extract from individuals
    for ind in onto.individuals():
        if hasattr(ind, rxnorm_prop.name):
            for rid in getattr(ind, rxnorm_prop.name):
                rxnorm_ids.add(str(rid))
    
    rxnorm_ids_list = list(rxnorm_ids)
    print(f"Found {len(rxnorm_ids)} unique RxNorm CUIs in VO")
    
    # 3. Create DataFrame of existing RxNorm terms with metadata
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
    
    vo_rxnorm_df = pd.DataFrame([get_rxnorm_data(rxcui) for rxcui in rxnorm_ids_list])
    
    print("Step 2: Fetching potential vaccine-related RxNorm concepts...")
    # 4. Fetch all potential vaccine-related RxNorm concepts
    
    # Helper functions
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
    
    # Collect RxNorm concepts from different sources
    rx_concepts = set()
    
    # ATC concepts
    print("Collecting from ATC...")
    atc_descendants = get_class_descendants("J07", "ATC1-4")
    for atc in atc_descendants:
        params = {"classId": atc['ID'], "relaSource": "ATCPROD"}
        res = get_rx_class_members(params)
        if res['related_ids']:
            rx_concepts.update(res['related_ids'])
    
    # VA concepts
    print("Collecting from VA...")
    VA_Classes = ["IM100", "IM105", "IM109"]
    rela = ["has_vaclass", "has_vaclass_extended"]
    for va in VA_Classes:
        for r in rela:
            params = {"classId": va, "relaSource": "VA", "rela": r}
            res = get_rx_class_members(params)
            if res['related_ids']:
                rx_concepts.update(res['related_ids'])
    
    # CVX concepts
    print("Collecting from CVX...")
    cvx_descendants = get_class_descendants("0", "CVX")
    for cvx in cvx_descendants:
        params = {"classId": cvx['ID'], "relaSource": "CDC", "rela": "isa_CVX"}
        res = get_rx_class_members(params)
        if res['related_ids']:
            rx_concepts.update(res['related_ids'])
    
    # DailyMed concepts
    print("Collecting from DailyMed...")
    dm_classes = ["N0000193912", "D014612"]
    rela = ["has_epc", "has_chemical_structure"]
    for dm in dm_classes:
        for r in rela:
            params = {"classId": dm, "relaSource": "DAILYMED", "rela": r}
            res = get_rx_class_members(params)
            if res['related_ids']:
                rx_concepts.update(res['related_ids'])
    
    print(f"Found {len(rx_concepts)} initial RxNorm concepts")
    
    # Get related concepts
    print("Finding related concepts...")
    all_rx_list = list(rx_concepts)
    count = 0
    for rxcui in rx_concepts:
        res = get_related_rxcui(rxcui)
        if res['related_ids']:
            all_rx_list.extend(res['related_ids'])
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{len(rx_concepts)} concepts...")
    
    all_rx_final = list(set(all_rx_list))
    print(f"Found {len(all_rx_final)} total RxNorm concepts after expansion")
    
    # 5. Compare with existing VO annotations
    print("Step 3: Identifying missing RxNorm terms...")
    rxnav_data = [get_rxnorm_data(rxcui) for rxcui in all_rx_final]
    rxnav_df = pd.DataFrame(rxnav_data)
    
    # Ensure both dataframes have rxcui as integers for comparison
    rxnav_df['rxcui'] = rxnav_df['rxcui'].astype(int)
    vo_rxnorm_df['rxcui'] = vo_rxnorm_df['rxcui'].astype(int)
    
    # Find missing terms
    missing_terms_df = rxnav_df[~rxnav_df['rxcui'].isin(vo_rxnorm_df['rxcui'])]
    
    print(f"Found {len(missing_terms_df)} RxNorm terms not in VO")
    
    return missing_terms_df

if __name__ == "__main__":
    identify_missing_rxnorm_terms()