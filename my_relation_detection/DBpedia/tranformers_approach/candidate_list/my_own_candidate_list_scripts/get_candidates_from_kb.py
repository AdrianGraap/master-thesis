from SPARQLWrapper import SPARQLWrapper, JSON
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql"
WIKIDATA_ENDPOINT = "https://wikidata.demo.openlinksw.com/sparql"


def get_all_candidates(entity, kb):
    resp = ''
    entity = entity.replace("'", "\\'").replace('"', '')
    query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX bif: <http://www.openlinksw.com/schemas/bif#>
            SELECT DISTINCT ?rel WHERE {{
                ?s rdfs:label ?label .
                ?s ?rel ?x .
                FILTER ( LANG(?label) = "en" )
                ?label bif:contains '"{entity}"'
            }}
            """
    if kb == 'dbpedia':
        sparql = SPARQLWrapper(DBPEDIA_ENDPOINT)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        for _ in range(10):
            try:
                response = sparql.query().convert()  # answer from the SPARQL Endpoint of a KG
                resp = response['results']['bindings']
                break
            # except SPARQLWrapper.SPARQLExceptions.EndPointInternalError:
            except:
                print('wiederholen')
    else:
        sparql = SPARQLWrapper(WIKIDATA_ENDPOINT)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        for _ in range(10):
            try:
                response = sparql.query().convert()  # answer from the SPARQL Endpoint of a KG
                resp = response['results']['bindings']
                break
            except:
                print('wiederholen')
    if resp:
        return [r['rel']['value'].replace('http://dbpedia.org/property/', 'dbp:')
                .replace('http://dbpedia.org/ontology/', 'dbo:')
                .replace('http://www.wikidata.org/prop/direct-normalized/', '')
                .replace('http://www.wikidata.org/prop/direct/', '')
                .replace('http://www.wikidata.org/prop/', '') for r in resp if kb in r['rel']['value']]
    else:
        return []
