# from tqdm import tqdm
# import numpy as np
# embedding_vector = {}
# f = open('eda_code/glove.840B.300d.txt')
# for line in tqdm(f):
#     value = line.split(' ')
#     word = value[0]
#     coef = np.array(value[1:],dtype = 'float32')
#     embedding_vector[word] = coef
# f.close()
# embedding_vector_list = list(embedding_vector.items())
# print(embedding_vector_list[5000][0])
# print(embedding_vector_list[5000][1])
# print(len(embedding_vector_list[5000][1]))
# print('len of list ==> ' + str(len(embedding_vector_list)))
# print('len of dict ==> ' + str(len(embedding_vector)))
# print('shape of list ==> ' + str(np.asarray(embedding_vector_list[5000],dtype=object).shape))
# print('shape of dict ==> ' + str(np.asarray(embedding_vector,dtype=object).shape))

from sklearn.manifold import TSNE
import plotly.express as px


df = px.data.iris()

features = df.loc[:, :'petal_width']

tsne = TSNE(n_components=2, random_state=0)
projections = tsne.fit_transform(features)

fig = px.scatter(
    projections, x=0, y=1,
    color=df.species, labels={'color': 'species'}
)
print('len of projection: ',len(projections))
print('shape of projection: ',projections.shape)
print(projections)

