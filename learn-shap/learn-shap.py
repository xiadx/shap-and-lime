#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
"""learn-shap"""


# In[2]:


# package
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# load data
data = pd.read_csv('train.csv')
data.head()


# In[7]:


# age
today = pd.to_datetime('2018-01-01')
data['birth_date'] = pd.to_datetime(data['birth_date'])
data['age'] = np.round((today - data['birth_date']).apply(lambda x: x.days) / 365., 1)
data.head()


# In[8]:


# feature
cols = ['height_cm', 'potential', 'pac', 'sho', 'pas', 'dri', 'def', 'phy', 'international_reputation', 'age']


# In[9]:


# xgboost regression model
model = xgb.XGBRegressor(max_depth=4, learning_rate=0.05, n_estimators=150)
model.fit(data[cols], data['y'].values)


# In[10]:


# feature importance
plt.figure(figsize=(15, 5))
plt.bar(range(len(cols)), model.feature_importances_)
plt.xticks(range(len(cols)), cols, rotation=-45, fontsize=14)
plt.title('Feature importance', fontsize=14)
plt.show()


# In[11]:


# shap
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data[cols])
print(shap_values.shape)


# In[12]:


y_base = explainer.expected_value
print(y_base)
data['pred'] = model.predict(data[cols])
print(data['pred'].mean())
data.head()


# In[13]:


j = 30
player_explainer = pd.DataFrame()
player_explainer['feature'] = cols
player_explainer['feature_value'] = data[cols].iloc[j].values
player_explainer['shap_value'] = shap_values[j]
player_explainer


# In[14]:


print('y_base + sum_of_shap_values: %.2f'%(y_base + player_explainer['shap_value'].sum()))
print('y_pred: %.2f'%(data['pred'].iloc[j]))


# In[15]:


shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[j], data[cols].iloc[j])


# In[27]:


shap.force_plot(explainer.expected_value, shap_values[:500], data[cols][:500])


# In[28]:


shap.summary_plot(shap_values, data[cols])


# In[29]:


shap.summary_plot(shap_values, data[cols], plot_type="bar")


# In[30]:


shap.dependence_plot('age', shap_values, data[cols], interaction_index=None, show=False)


# In[31]:


shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(data[cols])
shap.summary_plot(shap_interaction_values, data[cols], max_display=4)


# In[32]:


shap.dependence_plot('potential', shap_values, data[cols], interaction_index='international_reputation', show=False)


# In[35]:


# !jupyter nbconvert --to python .ipynb


# In[ ]:




