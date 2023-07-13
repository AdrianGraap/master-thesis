import os
import ssl

from SPARQLWrapper import SPARQLWrapper, JSON

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

def get_superclass(relation):
    dbpedia_query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
        select ?x where {{<{relation}> rdfs:subPropertyOf ?x}} 
        """

    dbpedia_endpoint = "https://dbpedia.org/sparql"

    sparql = SPARQLWrapper(dbpedia_endpoint)
    sparql.setQuery(dbpedia_query)
    sparql.setReturnFormat(JSON)
    response = sparql.query().convert()

    try:
        answer = response['results']['bindings'][0]['x']['value']
    except:
        answer = ''

    return answer
