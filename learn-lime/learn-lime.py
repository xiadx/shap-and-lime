#!/usr/bin/env python
# coding: utf-8

# In[20]:


#!/usr/bin/env python
"""learn lime"""


# In[21]:


# package
# from __future__ import print_function
import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics


# In[22]:


# fetching data
from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']


# In[23]:


# vectorizer
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)


# In[24]:


# fit
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, newsgroups_train.target)


# In[25]:


# predict
pred = rf.predict(test_vectors)
sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')


# In[26]:


# lime
from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, rf)


# In[27]:


# lime_text
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)


# In[28]:


# explain_instance
idx = 83
exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6)
print('Document id: %d' % idx)
print('Probability(christian) =', c.predict_proba([newsgroups_test.data[idx]])[0,1])
print('True class: %s' % class_names[newsgroups_test.target[idx]])


# In[29]:


exp.as_list()


# In[30]:


print('Original prediction:', rf.predict_proba(test_vectors[idx])[0,1])
tmp = test_vectors[idx].copy()
tmp[0,vectorizer.vocabulary_['Posting']] = 0
tmp[0,vectorizer.vocabulary_['Host']] = 0
print('Prediction removing some features:', rf.predict_proba(tmp)[0,1])
print('Difference:', rf.predict_proba(tmp)[0,1] - rf.predict_proba(test_vectors[idx])[0,1])


# In[31]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = exp.as_pyplot_figure()


# In[32]:


exp.show_in_notebook(text=False)


# In[ ]:


# exp.save_to_file('/tmp/oi.html')


# In[33]:


exp.show_in_notebook(text=True)


# In[34]:


# package
# from __future__ import print_function
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
np.random.seed(1)


# In[35]:


# iris
iris = sklearn.datasets.load_iris()


# In[36]:


# train_test_split
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(iris.data, iris.target, train_size=0.80)


# In[37]:


# fit
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train, labels_train)


# In[38]:


sklearn.metrics.accuracy_score(labels_test, rf.predict(test))


# In[39]:


# explainer
explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=True)


# In[40]:


# explain_instance
i = np.random.randint(0, test.shape[0])
exp = explainer.explain_instance(test[i], rf.predict_proba, num_features=2, top_labels=1)


# In[41]:


# show
exp.show_in_notebook(show_table=True, show_all=False)


# In[42]:


feature_index = lambda x: iris.feature_names.index(x)


# In[47]:


print('Increasing petal width')
temp = test[i].copy()
print('P(setosa) before:', rf.predict_proba(temp.reshape(1,-1))[0,0])
temp[feature_index('petal width (cm)')] = 1.5
print('P(setosa) after:', rf.predict_proba(temp.reshape(1,-1))[0,0])
print ()
print('Increasing petal length')
temp = test[i].copy()
print('P(setosa) before:', rf.predict_proba(temp.reshape(1,-1))[0,0])
temp[feature_index('petal length (cm)')] = 3.5
print('P(setosa) after:', rf.predict_proba(temp.reshape(1,-1))[0,0])
print()
print('Increasing both')
temp = test[i].copy()
print('P(setosa) before:', rf.predict_proba(temp.reshape(1,-1))[0,0])
temp[feature_index('petal width (cm)')] = 1.5
temp[feature_index('petal length (cm)')] = 3.5
print('P(setosa) after:', rf.predict_proba(temp.reshape(1,-1))[0,0])


# In[44]:


exp.show_in_notebook(show_table=True, show_all=True)


# In[48]:


feature_names = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status","Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country"]


# In[50]:


data = np.genfromtxt('adult.data', delimiter=', ', dtype=str)


# In[51]:


labels = data[:,14]
le= sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_
data = data[:,:-1]


# In[52]:


categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]


# In[53]:


categorical_names = {}
for feature in categorical_features:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data[:, feature])
    data[:, feature] = le.transform(data[:, feature])
    categorical_names[feature] = le.classes_


# In[54]:


data = data.astype(float)


# In[55]:


encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=categorical_features)


# In[57]:


np.random.seed(1)
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80)


# In[58]:


encoder.fit(data)
encoded_train = encoder.transform(train)


# In[59]:


import xgboost
gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
gbtree.fit(encoded_train, labels_train)


# In[60]:


sklearn.metrics.accuracy_score(labels_test, gbtree.predict(encoder.transform(test)))


# In[61]:


predict_fn = lambda x: gbtree.predict_proba(encoder.transform(x)).astype(float)


# In[62]:


explainer = lime.lime_tabular.LimeTabularExplainer(train ,feature_names = feature_names,class_names=class_names,
                                                   categorical_features=categorical_features, 
                                                   categorical_names=categorical_names, kernel_width=3)


# In[63]:


np.random.seed(1)
i = 1653
exp = explainer.explain_instance(test[i], predict_fn, num_features=5)
exp.show_in_notebook(show_all=False)


# In[64]:


i = 10
exp = explainer.explain_instance(test[i], predict_fn, num_features=5)
exp.show_in_notebook(show_all=False)


# In[ ]:




