import os
import ssl

from SPARQLWrapper import SPARQLWrapper, JSON

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


def get_label(url):
    dbpedia_query = f"""
    select ?x where {{<{url}> rdfs:label ?x FILTER (lang(?x) = 'en')}} 
    """

    dbpedia_endpoint = "https://dbpedia.org/sparql"

    sparql = SPARQLWrapper(dbpedia_endpoint)
    sparql.setQuery(dbpedia_query)
    sparql.setReturnFormat(JSON)
    response = sparql.query().convert()

    label = response['results']['bindings'][0]['x']['value']
    return label


def get_all_relations(entity):
    dbpedia_query = f"""
    PREFIX dbr: <http://dbpedia.org/resource/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT DISTINCT ?p 
    WHERE {{
      {{
        <{entity}> ?p ?o1 .
        FILTER( regex(STR(?p), "dbpedia.org/" ))
      }} 
      UNION
      {{
        ?o2 ?p <{entity}> .
        FILTER( regex(STR(?p), "dbpedia.org/" ))
      }}
    }}
    """

    all_relations = []

    dbpedia_endpoint = "https://dbpedia.org/sparql"

    sparql = SPARQLWrapper(dbpedia_endpoint)
    sparql.setQuery(dbpedia_query)
    sparql.setReturnFormat(JSON)
    response = sparql.query().convert()

    # print(response)

    for line in response['results']['bindings']:
        all_relations.append(line['p']['value'])
    return all_relations
