import json
import re


def check(question, query):
    if check_count(query):
        return 'count'
    elif check_boolean(query):
        return 'boolean'
    elif check_factoid(question):
        return 'factoid'
    else:
        return 'list'


def check_count(query):
    regex = r"SELECT.*COUNT.*{.*}"
    #return re.match(regex, query)
    return re.findall(regex, query)


def check_boolean(query):
    regex = r"ASK.*{.*}"
    return re.findall(regex, query)


def check_factoid(question):
    word_list = ['what', 'when', 'which', 'who', 'how', 'where', 'whom', 'whose', 'why']
    return (question.split(' ')[0].lower() in word_list) or (question.split(' ')[1].lower() in word_list)

# x = check('How many rivers and lakes are in South Carolina?', 'PREFIX dbo: <http://dbpedia.org/ontology/> PREFIX dct: <http://purl.org/dc/terms/> PREFIX dbc: <http://dbpedia.org/resource/Category:> PREFIX dbr: <http://dbpedia.org/resource/> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> SELECT (COUNT(DISTINCT ?uri) AS ?count) WHERE { { ?uri dbo:location dbr:South_Carolina ; rdf:type dbo:Lake } UNION { ?uri dct:subject dbc:Rivers_and_streams_of_South_Carolina } }')
# print(x)
