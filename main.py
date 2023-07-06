from pymilvus import connections, Collection
from text2vec import SentenceModel
from fastapi import FastAPI
from pydantic import BaseModel
import typing

app = FastAPI()
model = SentenceModel('shibing624/text2vec-base-chinese')
connections.connect(host='127.0.0.1', port='19530')
collection = Collection(name='question_answer')
search_params = {"metric_type": "L2", "params": {"nprobe": 10}, "offset": 5}


class RelateQa(BaseModel):
    id: str
    question: str
    answer: str
    distance: float


class Qa(BaseModel):
    question: str
    answer: str


def results2rqa(results):
    rqa_list = []
    ids = results[0].ids
    distances = results[0].distances
    for idx, distance, result in zip(ids, distances, results[0]):
        entity = result.entity
        rqa = {
            "id": idx,
            "question": entity.get('question'),
            "answer": entity.get('answer'),
            "distance": distance
        }
        rqa_list.append(rqa)
    return rqa_list


@app.get('/')
async def index():
    return "Hello World"


@app.get('/query', response_model=typing.List[RelateQa])
def query(question: str, limit: int = 3):
    collection.load()
    embedding = model.encode(question)
    results = collection.search(
        data=[embedding],
        anns_field='embedding',
        param=search_params,
        output_fields=['question', 'answer'],
        limit=limit,
        consistency_level="Strong"
    )
    collection.release()
    return results2rqa(results)


@app.get('/list_qa', response_model=typing.List[Qa])
def list_qa(offset: int = 0, limit=15):
    collection.load()
    results = collection.query(
        expr="id > 0",
        offset=offset,
        limit=limit,
        output_fields=["question", "answer"],
        consistency_level="Strong"
    )
    collection.release()
    return results


@app.put('/add_qa')
async def add_qa(question: str, answer: str):
    collection.load()
    mr = collection.insert([
        [question], [model.encode(question)], [answer]
    ])
    collection.release()
    return {"status": "succeed", "msg": "添加成功"}


@app.put('/add_qas')
async def add_qas(qas: typing.List[Qa], batch_size: int = 16):
    if batch_size > 64:
        return {"status": "failed", "msg": "batch_size不能超过64"}
    collection.load()
    for i in range(0, len(qas), batch_size):
        end = (i + batch_size) if (i + batch_size) <= len(qas) else -1
        batch = qas[i: end]
        qlist = list(map(lambda x: x.question, batch))
        alist = list(map(lambda x: x.answer, batch))
        embeddings = model.encode(qlist)
        mr = collection.insert([
            qlist,
            embeddings,
            alist
        ])
    collection.release()
    return {"status": "succeed", "msg": "添加成功"}
