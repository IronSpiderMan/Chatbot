import pandas as pd
from tqdm import tqdm
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from text2vec import SentenceModel

df = pd.read_csv('question_answer.csv')
id_answer = df.set_index('id')['answer'].to_dict()
model = SentenceModel('shibing624/text2vec-base-chinese')
connections.connect(host='127.0.0.1', port='19530')


def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, descrition='ids', max_length=500, is_primary=True,
                    auto_id=True),
        FieldSchema(name='question', dtype=DataType.VARCHAR, description='question content', max_length=512),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=dim),
        FieldSchema(name='answer', dtype=DataType.VARCHAR, description='answer content', max_length=512),
    ]
    schema = CollectionSchema(fields=fields, description='reverse image search')
    collection = Collection(name=collection_name, schema=schema)

    # create IVF_FLAT index for collection.
    index_params = {
        'metric_type': 'L2',
        'index_type': "IVF_FLAT",
        'params': {"nlist": 2048}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


collection = create_milvus_collection('question_answer', 768)
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}

pbar = tqdm(total=len(df.index))
for rid, row in df.iterrows():
    idx, q, a = row.values
    if len(q) > 512:
        q = q[:512]
    if len(a) > 512:
        a = a[:512]
    collection.insert([[q], [model.encode(q)], [a]])
    pbar.update(1)
print('Total number of inserted data is {}.'.format(collection.num_entities))
# search_params = {"metric_type": "L2", "params": {"nprobe": 10}, "offset": 5}
# while True:
#     question = input("请输入你的问题：")
#     if question.lower() == 'q':
#         break
#     embedding = model.encode(question)
#     results = collection.search(
#         data=[embedding],
#         anns_field='embedding',
#         param=search_params,
#         output_fields=['question'],
#         limit=5,
#         consistency_level="Strong"
#     )
#     print(results[0].ids)
