#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
"""xgboost shap and lime"""


# In[2]:


# parameter
MODEL = "onTravelV6C"
N_SAMPLES = 500
TRAIN_DATA_FILE = "train_" + MODEL + ".txt"
SAMPLE_FILE = "sample_train_" + MODEL + ".txt"
FEATURE_MAP_FILE = "feature_map_" + MODEL + ".json"
MODEL_FILE = MODEL + ".bin"
SAMPLE_FILE = "sample_" + str(N_SAMPLES) + "_" + TRAIN_DATA_FILE


# In[3]:


get_ipython().run_cell_magic('bash', '', '# prepare\n\n# parameter\nMODEL="onTravelV6C"\nN_SAMPLES=500\nTRAIN_DATA_FILE="train_${MODEL}.txt"\nSAMPLE_FILE="sample_train_${MODEL}.txt"\nFEATURE_MAP_FILE="feature_map_${MODEL}.json"\nMODEL_FILE="${MODEL}.bin"\nSAMPLE_FILE="sample_${N_SAMPLES}_${TRAIN_DATA_FILE}"\n\n# train data file\nif [[ ! -f ${TRAIN_DATA_FILE} ]]; then\n    echo "Train Data File Not Exist"\n    echo "Copy File Begin"\n    cp /mfw_data/algo/wanglei/spark_offline/train_data/onTravel/${TRAIN_DATA_FILE} ./\n    echo "Copy File End"\nfi\n\n# feature map data file\nif [[ ! -f ${FEATURE_MAP_FILE} ]]; then\n    echo "Feature Map File Not Exist"\n    echo "Get File Begin"\n    hadoop fs -text /user/wanglei3/featureMap/onTravel/${MODEL}/part-00000.snappy > ${FEATURE_MAP_FILE}\n    echo "Get File End"\nfi\n\n# xgboost model file\nif [[ ! -f ${MODEL_FILE} ]]; then\n    echo "Model File Not Exist"\n    echo "Copy File Begin"\n    cp /opt/tomcat/webapps/model/${MODEL} ./\n    mv ${MODEL} ${MODEL}.bin\n    echo "Copy File End"\nfi\n\n# random sampling\nif [[ ! -f ${SAMPLE_FILE} ]]; then\n    echo "Sample File Not Exist"\n    echo "Sampling Begin"\n    shuf -n ${N_SAMPLES} ${TRAIN_DATA_FILE} -o sample_${N_SAMPLES}_${TRAIN_DATA_FILE}\n    echo "Sampling End"\nfi\n\nls')


# In[4]:


# ipython core option  
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[5]:


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


# In[6]:


# feature map
with open(FEATURE_MAP_FILE) as fp:
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


# In[7]:


# load libsvm format file
X, y = load_svmlight_file(SAMPLE_FILE, n_features=len(cols))
print(X[0].todense().shape)
print(y[0])


# In[8]:


# create dataframe
df = pd.DataFrame(X.todense())
df.columns = cols
df["repair"] = np.zeros(N_SAMPLES)
df["label"] = y
df = df[["repair"]+cols+["label"]]
df.head()


# In[9]:


IS_TRAIN = False


# In[10]:


# train xgboost model
if IS_TRAIN:
#     from sklearn.ensemble import GradientBoostingClassifier
#     param = {
#         "loss": "deviance",
#         "learning_rate": 0.1,
#         "max_depth": 7,
#         "subsample": 0.8,
#         "n_estimators": 300
#     }
#     sk_gbt = GradientBoostingClassifier(**param)
#     sk_gbt.fit(df[["repair"]+cols], df["label"])

#     param = {
#         "objective": "binary:logistic",
#         "learning_rate": 0.1,
#         "max_depth": 7,
#         "min_child_weight": 1,
#         "gamma": 0,
#         "subsample": 0.8,
#         "colsample_bytree": 0.8,
#         "scale_pos_weight": 1,
#         "n_estimators": 300,
#     }
#     sk_xgb = xgb.XGBClassifier(**param)
#     sk_xgb.fit(df[["repair"]+cols], df["label"])

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


# In[11]:


if IS_TRAIN == True:
    model = bst_xgb
else:
    model = bst


# In[16]:


# margin or probability
MODEL_OUTPUT = "probability"


# In[17]:


# shap
if MODEL_OUTPUT == "margin":
    # margin explanation
    shap_explainer = shap.TreeExplainer(model)
if MODEL_OUTPUT == "probability":
    # probability explanation
    BACKGROUND_DATASET_SIZE = 1000
    if len(df[["repair"]+cols]) <= BACKGROUND_DATASET_SIZE:
        background_dataset = df[["repair"]+cols]
    else:
        background_dataset = df[["repair"]+cols].sample(BACKGROUND_DATASET_SIZE)
    shap_explainer = shap.TreeExplainer(model, background_dataset.values, model_output="probability", feature_dependence="independent")


# In[20]:


shap_values = shap_explainer.shap_values(df[["repair"]+cols])
print("shap_values: ", shap_values.shape)
y_base = shap_explainer.expected_value
print("y_base: ", y_base)


# In[22]:


if MODEL_OUTPUT == "margin":
    # margin explanation
    df["pred"] = model.predict(xgb.DMatrix(df[["repair"]+cols], label=df["label"]), output_margin=True)
if MODEL_OUTPUT == "probability":
    # probability explanation
    df["pred"] = model.predict(xgb.DMatrix(df[["repair"]+cols], label=df["label"]), output_margin=False)
print("pred mean: ", df["pred"].mean())
df.head()


# In[23]:


shap.force_plot(shap_explainer.expected_value, shap_values, df[["repair"]+cols])


# In[24]:


shap.summary_plot(shap_values, df[["repair"]+cols], plot_type="bar")


# In[25]:


shap.summary_plot(shap_values, df[["repair"]+cols])


# In[26]:


if MODEL_OUTPUT == "margin":
    shap_interaction_values = shap_explainer.shap_interaction_values(df[["repair"]+cols])
    shap.summary_plot(shap_interaction_values, df[["repair"]+cols], max_display=4)


# In[27]:


# j = np.random.randint(N_SAMPLES)


# In[34]:


i = np.random.choice(df[df["pred"] <= 0.5].index.tolist())
print("negative sample")
player_explainer = pd.DataFrame()
player_explainer['feature'] = ["repair"]+cols
player_explainer['feature_value'] = df[["repair"]+cols].iloc[i].values
player_explainer['shap_value'] = shap_values[i]
player_explainer
print("y_base + sum_of_shap_values: %.2f" % (y_base + player_explainer["shap_value"].sum()))
print("y_pred: %.2f" % (df["pred"].iloc[i]))


# In[29]:


shap.initjs()
shap.force_plot(shap_explainer.expected_value, shap_values[i], df[["repair"]+cols].iloc[i])


# In[35]:



j = np.random.choice(df[df["pred"] >= 0.5].index.tolist())
print("positive sample")
player_explainer = pd.DataFrame()
player_explainer['feature'] = ["repair"]+cols
player_explainer['feature_value'] = df[["repair"]+cols].iloc[j].values
player_explainer['shap_value'] = shap_values[j]
player_explainer
print("y_base + sum_of_shap_values: %.2f" % (y_base + player_explainer["shap_value"].sum()))
print("y_pred: %.2f" % (df["pred"].iloc[j]))


# In[36]:


shap.initjs()
shap.force_plot(shap_explainer.expected_value, shap_values[j], df[["repair"]+cols].iloc[j])


# In[38]:


FEATURE="doubleFlow_article_ctr_30_v1"
INTERACTION="doubleFlow_user_view_30"
shap.dependence_plot(FEATURE, shap_values, df[["repair"]+cols], interaction_index=None, show=False)
shap.dependence_plot(FEATURE, shap_values, df[["repair"]+cols], interaction_index=INTERACTION, show=False) 


# In[39]:


# lime
lime_explainer = lime.lime_tabular.LimeTabularExplainer(df[["repair"]+cols].values, 
                                                   feature_names=["repair"]+cols,
                                                   class_names=["0", "1"], 
                                                   verbose=True)


# In[40]:


model.feature_names = None
def predict_fn(x):
    preds = model.predict(xgb.DMatrix(x))
    return np.array([[1-p, p] for p in preds])


# In[41]:


i = np.random.choice(df[df["pred"] <= 0.5].index.tolist())
print("negative sample")
player_explainer = pd.DataFrame()
player_explainer['feature'] = ["repair"]+cols
player_explainer['feature_value'] = df[["repair"]+cols].iloc[i].values
player_explainer['shap_value'] = shap_values[i]
player_explainer


# In[43]:


exp = lime_explainer.explain_instance(df[["repair"]+cols].values[i], predict_fn, num_features=5)
exp.show_in_notebook(show_table=True)


# In[44]:


exp.as_list()
fig = exp.as_pyplot_figure()
fig.show()


# In[46]:


j = np.random.choice(df[df["pred"] >= 0.5].index.tolist())
print("positive sample")
player_explainer = pd.DataFrame()
player_explainer['feature'] = ["repair"]+cols
player_explainer['feature_value'] = df[["repair"]+cols].iloc[j].values
player_explainer['shap_value'] = shap_values[j]
player_explainer


# In[47]:


exp = lime_explainer.explain_instance(df[["repair"]+cols].values[j], predict_fn, num_features=5)
exp.show_in_notebook(show_table=True)


# In[48]:


exp.as_list()
fig = exp.as_pyplot_figure()
fig.show()


# In[ ]:




