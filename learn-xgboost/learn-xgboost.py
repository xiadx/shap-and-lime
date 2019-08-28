#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
"""learn xgboost"""


# In[3]:


TRAIN_FILE = "agaricus.txt.train"
TEST_FILE = "agaricus.txt.test"
FEATMAP_FILE = "featmap.txt"


# In[4]:


# ipython core option  
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[5]:


# featmap
import re
with open(FEATMAP_FILE) as fp:
    cols = []
    for line in fp:
        cols.append(re.search(r"\t(.*)\t", line).group(1))
print(len(cols))
cols


# In[6]:


# load libsvm format file
from sklearn.datasets import load_svmlight_file
X, y = load_svmlight_file(TRAIN_FILE, n_features=len(cols))
print(type(X))
print(type(X.todense()))
print(type(X.toarray()))
print(X.todense().shape)


# In[7]:


# dump libsvm format file
from sklearn.datasets import dump_svmlight_file
DUMP_LIBSVM_FILE = "agaricus.txt.train.dump"
dump_svmlight_file(X, y, DUMP_LIBSVM_FILE, zero_based=True)
get_ipython().system('ls')


# In[8]:


# create dataframe
import pandas as pd
df = pd.DataFrame(X.toarray())
df.columns = cols
df["label"] = y
df.head()


# In[9]:


# train
from sklearn.ensemble import GradientBoostingClassifier
param = {
    "loss": "deviance",
    "learning_rate": 0.3,
    "max_depth": 2,
    "subsample": 0.8,
    "n_estimators": 5
}
sk_gbt = GradientBoostingClassifier(**param)
sk_gbt.fit(df[cols], df["label"])


# In[10]:


# load libsvm format file
from sklearn.datasets import load_svmlight_file
X, y = load_svmlight_file(TEST_FILE, n_features=len(cols))
print(type(X))
print(type(X.todense()))
print(type(X.toarray()))
print(X.todense().shape)


# In[11]:


# create dataframe
import pandas as pd
df = pd.DataFrame(X.toarray())
df.columns = cols
df["label"] = y
df.head()


# In[12]:


# test
y_pred = [x[1] for x in sk_gbt.predict_proba(df[cols])]
df["pred"] = y_pred
df.head()


# In[13]:


# auc
def auc(y_true, y_pred):
    """
    calculate auc
    Args:
        y_true: label
        y_pred: predict
    Return:
        auc
    """
    l = [(i, t[0], t[1]) for i, t in enumerate(sorted(zip(y_true, y_pred), key=lambda x: x[1]))]
    # p = sum([x[1] for x in l])
    # n = len(l) - p
    # t = sum([x[0] for x in l if x[1] == 1])
    n, p, t = 0, 0, 0
    for i, y, r in l:
        if y == 0:
            n += 1
        else:
            p += 1
            t += i
    return (t-(p*(p+1)/2)) / (n*p)
from sklearn.metrics import roc_auc_score
y_true = df["label"].values
y_pred = df["pred"].values
print(roc_auc_score(y_true, y_pred))
print(auc(y_true, y_pred))


# In[14]:


# load libsvm format file
from sklearn.datasets import load_svmlight_file
X, y = load_svmlight_file(TRAIN_FILE, n_features=len(cols))
print(type(X))
print(type(X.todense()))
print(type(X.toarray()))
print(X.todense().shape)


# In[15]:


# create dataframe
import pandas as pd
df = pd.DataFrame(X.toarray())
df.columns = cols
df["label"] = y
df.head()


# In[16]:


# train
import xgboost as xgb
param = {
    "objective": "binary:logistic",
    "learning_rate": 0.3,
    "max_depth": 2,
    "min_child_weight": 1,
    "gamma": 0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1,
    "n_estimators": 5,
}
sk_xgb = xgb.XGBClassifier(**param)
sk_xgb.fit(df[cols], df["label"])


# In[17]:


# load libsvm format file
from sklearn.datasets import load_svmlight_file
X, y = load_svmlight_file(TEST_FILE, n_features=len(cols))
print(type(X))
print(type(X.todense()))
print(type(X.toarray()))
print(X.todense().shape)


# In[18]:


# create dataframe
import pandas as pd
df = pd.DataFrame(X.toarray())
df.columns = cols
df["label"] = y
df.head()


# In[19]:


# test
y_pred = [x[1] for x in sk_gbt.predict_proba(df[cols])]
df["pred"] = y_pred
df.head()


# In[20]:


# auc
def auc(y_true, y_pred):
    """
    calculate auc
    Args:
        y_true: label
        y_pred: predict
    Return:
        auc
    """
    l = [(i, t[0], t[1]) for i, t in enumerate(sorted(zip(y_true, y_pred), key=lambda x: x[1]))]
    # p = sum([x[1] for x in l])
    # n = len(l) - p
    # t = sum([x[0] for x in l if x[1] == 1])
    n, p, t = 0, 0, 0
    for i, y, r in l:
        if y == 0:
            n += 1
        else:
            p += 1
            t += i
    return (t-(p*(p+1)/2)) / (n*p)
from sklearn.metrics import roc_auc_score
y_true = df["label"].values
y_pred = df["pred"].values
print(roc_auc_score(y_true, y_pred))
print(auc(y_true, y_pred))


# In[21]:


# load libsvm format file
from sklearn.datasets import load_svmlight_file
X, y = load_svmlight_file(TRAIN_FILE, n_features=len(cols))
print(type(X))
print(type(X.todense()))
print(type(X.toarray()))
print(X.todense().shape)


# In[22]:


# create dataframe
import pandas as pd
df = pd.DataFrame(X.toarray())
df.columns = cols
df["label"] = y
df.head()


# In[23]:


# train
param = {
        "objective": "binary:logistic",
        "eta": 0.3,
        "max_depth": 4,
        "min_child_weight": 1,
        "gamma": 0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1,
        "silent": True
    }
num_boost_round = 5
dtrain = xgb.DMatrix(df[cols], label=df["label"])
bst_xgb = xgb.train(param, dtrain, num_boost_round=num_boost_round)


# In[24]:


# load libsvm format file
from sklearn.datasets import load_svmlight_file
X, y = load_svmlight_file(TEST_FILE, n_features=len(cols))
print(type(X))
print(type(X.todense()))
print(type(X.toarray()))
print(X.todense().shape)


# In[25]:


# create dataframe
import pandas as pd
df = pd.DataFrame(X.toarray())
df.columns = cols
df["label"] = y
df.head()


# In[26]:


dtest = xgb.DMatrix(df[cols], label=df["label"])
p_pred = bst_xgb.predict(dtest)
df["pred"] = p_pred
df.head()


# In[27]:


# auc
def auc(y_true, y_pred):
    """
    calculate auc
    Args:
        y_true: label
        y_pred: predict
    Return:
        auc
    """
    l = [(i, t[0], t[1]) for i, t in enumerate(sorted(zip(y_true, y_pred), key=lambda x: x[1]))]
    # p = sum([x[1] for x in l])
    # n = len(l) - p
    # t = sum([x[0] for x in l if x[1] == 1])
    n, p, t = 0, 0, 0
    for i, y, r in l:
        if y == 0:
            n += 1
        else:
            p += 1
            t += i
    return (t-(p*(p+1)/2)) / (n*p)
from sklearn.metrics import roc_auc_score
y_true = df["label"].values
y_pred = df["pred"].values
print(roc_auc_score(y_true, y_pred))
print(auc(y_true, y_pred))


# In[28]:


# train
param = {
        "objective": "binary:logistic",
        "eta": 0.3,
        "max_depth": 4,
        "min_child_weight": 1,
        "gamma": 0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1,
        "silent": True
    }
num_boost_round = 5
dtrain = xgb.DMatrix(TRAIN_FILE)
bst = xgb.train(param, dtrain, num_boost_round=num_boost_round)


# In[29]:


dtest = xgb.DMatrix(TEST_FILE)
print(dtest.num_row())
print(dtest.num_col())
print(dtest.feature_names)


# In[33]:


# get_dump maybe dislocation problem
bst.get_dump()
bst.get_dump(FEATMAP_FILE)


# In[36]:


# get_score maybe dislocation problem
bst.get_score()
bst.get_score(FEATMAP_FILE)


# In[38]:


p_pred = bst.predict(dtest)
p_pred


# In[39]:


# auc
def auc(y_true, y_pred):
    """
    calculate auc
    Args:
        y_true: label
        y_pred: predict
    Return:
        auc
    """
    l = [(i, t[0], t[1]) for i, t in enumerate(sorted(zip(y_true, y_pred), key=lambda x: x[1]))]
    # p = sum([x[1] for x in l])
    # n = len(l) - p
    # t = sum([x[0] for x in l if x[1] == 1])
    n, p, t = 0, 0, 0
    for i, y, r in l:
        if y == 0:
            n += 1
        else:
            p += 1
            t += i
    return (t-(p*(p+1)/2)) / (n*p)
from sklearn.metrics import roc_auc_score
y_true = df["label"].values
y_pred = df["pred"].values
print(roc_auc_score(y_true, y_pred))
print(auc(y_true, y_pred))


# In[45]:


get_ipython().run_line_magic('matplotlib', 'inline')
bst.feature_names = ["repair"]+cols
xgb.plot_importance(bst)


# In[47]:


# save_model
MODEL_FILE = "bst.model"
bst.save_model(MODEL_FILE)


# In[ ]:




