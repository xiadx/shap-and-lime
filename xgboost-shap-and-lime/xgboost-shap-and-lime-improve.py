#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
"""xgboost shap and lime improve"""


# In[2]:


# parameter
MODEL = "onTravelV6C"
TRAIN_SAMPLES = 5000
TEST_SAMPLES = 1000
TRAIN_DATA_FILE = "train_" + MODEL + ".txt"
TEST_DATA_FILE = "test_" + MODEL + ".txt"
TRAIN_SAMPLE_FILE = "sample_" + str(TRAIN_SAMPLES) + "_" + TRAIN_DATA_FILE
TEST_SAMPLE_FILE = "sample_" + str(TEST_SAMPLES) + "_" + TEST_DATA_FILE
FEATMAP_FILE = "feature_map_" + MODEL + ".json"
MODEL_FILE = MODEL + ".model"


# In[3]:


# package
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("seaborn")
import shap
import lime
import json
import re
import os
import sys


# In[4]:


# prepare
if not os.path.exists(MODEL_FILE):
    flag = os.system("cp /opt/tomcat/webapps/model/" + MODEL + " ./")
    status = os.system("mv " + MODEL + " " + MODEL_FILE)
    if flag != 0 or status != 0:
        print("get file failure")
        sys.exit(1)
if not os.path.exists(FEATMAP_FILE):
    flag = os.system("hadoop fs -text /user/wanglei3/featureMap/onTravel/" + MODEL + "/part-00000.snappy > " + FEATMAP_FILE)
    if flag != 0:
        print("get file failure")
        sys.exit(1)
if not os.path.exists(TRAIN_DATA_FILE):
    flag = os.system("cp /mfw_data/algo/wanglei/spark_offline/train_data/onTravel/" + TRAIN_DATA_FILE + " ./")
    if flag != 0:
        print("get file failure")
        sys.exit(1)
if not os.path.exists(TEST_DATA_FILE):
    flag = os.system("cp /mfw_data/algo/wanglei/spark_offline/train_data/onTravel/" + TEST_DATA_FILE + " ./")
    if flag != 0:
        print("get file failure")
        sys.exit(1)


# In[5]:


# sample
if not os.path.exists(TRAIN_SAMPLE_FILE):
    fr = open(TRAIN_DATA_FILE, "r")
    fw = open(TRAIN_SAMPLE_FILE, "w")
    i, j, n = 0, 0, int(TRAIN_SAMPLES / 2)
    for line in fr:
        label = float(line.strip().split()[0])
        if label == 0.0:
            if i < n:
                fw.write(line)
                i += 1
        else:
            if j < n:
                fw.write(line)
                j += 1
        if i >= n and j >= n:
            break
    fw.close()
    fr.close()
if not os.path.exists(TEST_SAMPLE_FILE):
    fr = open(TEST_DATA_FILE, "r")
    fw = open(TEST_SAMPLE_FILE, "w")
    i, j, n = 0, 0, int(TEST_SAMPLES / 2)
    for line in fr:
        label = float(line.strip().split()[0])
        if label == 0.0:
            if i < n:
                fw.write(line)
                i += 1
        else:
            if j < n:
                fw.write(line)
                j += 1
        if i >= n and j >= n:
            break
    fw.close()
    fr.close()


# In[6]:


get_ipython().system('ls')


# In[7]:


# feature map
with open(FEATMAP_FILE) as fp:
    feature_map = json.load(fp)
cols = []
i = 0
for fm in feature_map:
    if i == 0:
        pass
    else:
        print(fm)
        cols.append(re.search(r"\t(.*)\t", fm).group(1))
    i += 1


# In[8]:


# load libsvm format file
X, y = load_svmlight_file(TEST_SAMPLE_FILE, n_features=len(cols))
print(X.toarray().shape)


# In[9]:


# create dataframe
df = pd.DataFrame(X.toarray())
df.columns = cols
df["repair"] = np.zeros(TEST_SAMPLES)
df["label"] = y
df = df[["repair"]+cols+["label"]]
df.head()


# In[10]:


IS_TRAIN = False


# In[11]:


# train xgboost model
if IS_TRAIN:
    param = {
        "objective": "binary:logistic",
        "eta": 0.1,
        "max_depth": 7,
        "min_child_weight": 1,
        "gamma": 0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1,
        "silent": True
    }
    num_boost_round = 300
    dtrain = xgb.DMatrix(df[["repair"]+cols], label=df["label"])
    bst_xgb = xgb.train(param, dtrain, num_boost_round=num_boost_round)
else:
    bst = xgb.Booster(model_file=MODEL_FILE)


# In[12]:


if IS_TRAIN == True:
    model = bst_xgb
else:
    model = bst


# In[13]:


# margin or probability
MODEL_OUTPUT = "probability"


# In[14]:


# shap
if MODEL_OUTPUT == "margin":
    # margin explanation
    shap_explainer = shap.TreeExplainer(model)
if MODEL_OUTPUT == "probability":
    # probability explanation
    shap_explainer = shap.TreeExplainer(model, df[["repair"]+cols], model_output="probability", feature_dependence="independent")


# In[ ]:


shap_values = shap_explainer.shap_values(df[["repair"]+cols])
print("shap_values.shape: ", shap_values.shape)
y_base = shap_explainer.expected_value
print("y_base: ", y_base)


# In[ ]:


if MODEL_OUTPUT == "margin":
    # margin explanation
    df["pred"] = model.predict(xgb.DMatrix(df[["repair"]+cols], label=df["label"]), output_margin=True)
if MODEL_OUTPUT == "probability":
    # probability explanation
    df["pred"] = model.predict(xgb.DMatrix(df[["repair"]+cols], label=df["label"]), output_margin=False)
print("pred mean: ", df["pred"].mean())
df.head()


# In[ ]:


shap.initjs()
shap.force_plot(shap_explainer.expected_value, shap_values, df[["repair"]+cols])


# In[ ]:


shap.summary_plot(shap_values, df[["repair"]+cols], plot_type="bar")


# In[ ]:


import seaborn as sns
feature_importance = pd.DataFrame([(x[0], x[1]) for x in model.get_score().items()], columns=["feature", "importance"])
feature_importance.sort_values(by=["importance"], ascending=False, inplace=True)
sns.barplot(x="importance", y="feature", data=feature_importance[:20], color="dodgerblue")


# In[ ]:


shap.summary_plot(shap_values, df[["repair"]+cols])


# In[ ]:


if MODEL_OUTPUT == "margin":
    shap_interaction_values = shap_explainer.shap_interaction_values(df[["repair"]+cols])
    shap.summary_plot(shap_interaction_values, df[["repair"]+cols], max_display=4)


# In[ ]:


# j = np.random.randint(N_SAMPLES)


# In[ ]:


i = np.random.choice(df[df["pred"] <= y_base].index.tolist())
print("negative sample")
player_explainer = pd.DataFrame()
player_explainer['feature'] = ["repair"]+cols
player_explainer['feature_value'] = df[["repair"]+cols].iloc[i].values
player_explainer['shap_value'] = shap_values[i]
print("y_base + sum_of_shap_values: %.2f" % (y_base + player_explainer["shap_value"].sum()))
print("y_pred: %.2f" % (df["pred"].iloc[i]))


# In[ ]:


shap.initjs()
shap.force_plot(shap_explainer.expected_value, shap_values[i], df[["repair"]+cols].iloc[i])


# In[ ]:


j = np.random.choice(df[df["pred"] >= y_base].index.tolist())
print("positive sample")
player_explainer = pd.DataFrame()
player_explainer['feature'] = ["repair"]+cols
player_explainer['feature_value'] = df[["repair"]+cols].iloc[j].values
player_explainer['shap_value'] = shap_values[j]
print("y_base + sum_of_shap_values: %.2f" % (y_base + player_explainer["shap_value"].sum()))
print("y_pred: %.2f" % (df["pred"].iloc[j]))


# In[ ]:


shap.initjs()
shap.force_plot(shap_explainer.expected_value, shap_values[j], df[["repair"]+cols].iloc[j])


# In[ ]:


FEATURE="doubleFlow_article_ctr_30_v1"
INTERACTION="doubleFlow_user_view_30"
shap.dependence_plot(FEATURE, shap_values, df[["repair"]+cols], interaction_index=None, show=False)
shap.dependence_plot(FEATURE, shap_values, df[["repair"]+cols], interaction_index=INTERACTION, show=False) 


# In[ ]:


# lime
lime_explainer = lime.lime_tabular.LimeTabularExplainer(df[["repair"]+cols].values, 
                                                   feature_names=["repair"]+cols,
                                                   class_names=["0", "1"], 
                                                   verbose=True)


# In[ ]:


model.feature_names = None
def predict_fn(x):
    preds = model.predict(xgb.DMatrix(x))
    return np.array([[1-p, p] for p in preds])


# In[ ]:


print("negative sample")
i = np.random.choice(df[df["pred"] <= y_base].index.tolist())


# In[ ]:


exp = lime_explainer.explain_instance(df[["repair"]+cols].values[i], predict_fn, num_features=5)
exp.show_in_notebook(show_table=True)


# In[ ]:


exp.as_list()
fig = exp.as_pyplot_figure()
fig.show()


# In[ ]:


print("positive sample")
j = np.random.choice(df[df["pred"] >= y_base].index.tolist())


# In[ ]:


exp = lime_explainer.explain_instance(df[["repair"]+cols].values[j], predict_fn, num_features=5)
exp.show_in_notebook(show_table=True)


# In[ ]:


exp.as_list()
fig = exp.as_pyplot_figure()
fig.show()


# In[ ]:


get_ipython().system('jupyter nbconvert --to python xgboost-shap-and-lime-improve.ipynb')


# In[ ]:


get_ipython().system('jupyter nbconvert --to html xgboost-shap-and-lime-improve.ipynb')


# In[ ]:


# !jupyter nbconvert --to pdf xgboost-shap-and-lime-improve.ipynb


# In[ ]:


get_ipython().system('ls')

