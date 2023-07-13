FROM python:3.9 as install

COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt


FROM install as run_application

COPY convert.py .
COPY count_and_map.py .
COPY SMART2022-RL-dbpedia-relation-vocabulary.json .
COPY SMART2022-RL-dbpedia-train.json .

RUN mkdir "relation_prediction"
COPY create_model.py ./relation_prediction

#ONLY FOR SCRIPTS THAT USE CANDIDATE LIST
#COPY candidate_list candidate_list

WORKDIR relation_prediction

CMD ["python", "create_model.py"]