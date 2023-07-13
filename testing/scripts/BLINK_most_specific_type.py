import os
import ssl

from SPARQLWrapper import SPARQLWrapper, JSON

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


def get_most_specific_type(entity):
    dbpedia_query = f"""
    PREFIX dbr: <http://dbpedia.org/resource/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    select distinct ?type1 ?type2 where {{ 
    <{entity}> rdf:type ?type1 .
    <{entity}> rdf:type ?type2 .
    ?type1 rdfs:subClassOf ?type2 .
    FILTER( regex(STR(?type2), "dbpedia.org/ontology" ))
    }}
    """

    type1 = []
    type2 = []

    dbpedia_endpoint = "https://dbpedia.org/sparql"

    sparql = SPARQLWrapper(dbpedia_endpoint)
    sparql.setQuery(dbpedia_query)
    sparql.setReturnFormat(JSON)
    response = sparql.query().convert()

    # print(response)

    for line in response['results']['bindings']:
        type1.append(line['type1']['value'])
        type2.append(line['type2']['value'])

    searched_item = [element for element in type1 if element not in type2]

    try:
        searched_item = searched_item[0].split('/')[-1]
    except IndexError:
        raise

    return searched_item
