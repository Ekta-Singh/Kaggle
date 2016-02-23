
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# In[76]:

df_train=pd.read_csv('/Users/ekta/Desktop/Kaggle/Home Depot/train.csv')
df_product_desc=pd.read_csv('/Users/ekta/Desktop/Kaggle/Home Depot/product_descriptions.csv')
df_attribute=pd.read_csv('/Users/ekta/Desktop/Kaggle/Home Depot/attributes.csv')


# In[77]:

## Merging train and product description datasets
df_train_proddesc=pd.merge(df_train,df_product_desc,how='left',on='product_uid')


# In[78]:

df_train_proddesc.isnull().sum() 
df_train_proddesc.shape # (74067, 6) with no duplicates 


# In[79]:

## Coverting everything into lower case
df_attribute.name=df_attribute['name'].map(lambda x: str(x).lower())
df_attribute.value=df_attribute['value'].map(lambda x: str(x).lower())
df_train_proddesc.product_title=df_train_proddesc['product_title'].map(lambda x: str(x).lower())
df_train_proddesc.search_term=df_train_proddesc['search_term'].map(lambda x: str(x).lower())
df_train_proddesc.product_description=df_train_proddesc['product_description'].map(lambda x: str(x).lower())


# In[80]:

## Remove duplicates attribute table
df_attribute.shape # (2044803, 3) with 154 duplicates
df_attribute=df_attribute.drop_duplicates() # (2044373, 3)


# In[81]:

## Fill missing values with "missing"
df_attribute=df_attribute[df_attribute['product_uid'].isnull()==False]
df_attribute['name']=df_attribute['name'].fillna('missing') 
df_attribute['value']=df_attribute['value'].fillna('missing') 


# In[82]:

## Getting brand for each product
df_attribute_1=df_attribute[df_attribute['name'].str.contains('mfg brand')][['product_uid','value']]
df_train_proddesc_1=pd.merge(df_train_proddesc,df_attribute_1,how='left',on='product_uid')
df_train_proddesc_1=df_train_proddesc_1.rename(columns={'value':'brand'}) 


# In[83]:

productIDWithNoBrand=df_train_proddesc_1[df_train_proddesc_1['brand'].isnull()==True]['product_uid'].unique()
df_attribute[(df_attribute['product_uid'].isin(productIDWithNoBrand)) & (df_attribute['name'].str.contains('brand'))]


# In[84]:

## Getting product ids for which there are multiple material type fields in attribute data
repeatedMaterialIDs=pd.DataFrame({'count':df_attribute[df_attribute['name'].str.contains('material')].groupby(['product_uid']).size()}).reset_index()
productIDsWithMultipleMaterial=repeatedMaterialIDs[repeatedMaterialIDs['count']>1]['product_uid']
a=df_attribute[(df_attribute['product_uid'].isin(productIDsWithMultipleMaterial)) & (df_attribute['name'].str.contains('material'))]


# In[85]:

## Grouping products with multiple materials in one row
df_attribute_1=df_attribute[df_attribute['product_uid'].isin(productIDsWithMultipleMaterial) & df_attribute['name'].str.contains('material')]
df_attribute_material=pd.DataFrame(df_attribute_1.groupby('product_uid').agg({'value':lambda x:','.join(set(x))})).reset_index()
df_attribute_material['name']='material'
df_attribute_f=df_attribute[-df_attribute.index.isin(list(a.index))]
df_attribute_f=pd.concat([df_attribute_material,df_attribute_f])


# In[86]:

## Getting material column in train dataset
df_attribute_1=df_attribute_f[df_attribute_f['name'].str.contains('material')]
df_train_proddesc_2=pd.merge(df_train_proddesc_1,df_attribute_1,how='left',on='product_uid')
df_train_proddesc_2=df_train_proddesc_2.rename(columns={'value':'material'}) 


# In[15]:

df_attribute_1=df_attribute[df_attribute['name'].str.contains('Color')][['product_uid','value']]
df_train_proddesc_3=pd.merge(df_train_proddesc_2,df_attribute_1,how='left',on='product_uid')
df_train_proddesc_3=df_train_proddesc_3.rename(columns={'value':'Color'})


# In[90]:

#b=df_attribute[(df_attribute['name']=='color')|(df_attribute['name']=='color family')]
b=df_attribute[(df_attribute['name'].str.contains('color')]
# (33861,41508) 23840 color finish and color
# (28564, 4629) 3644 treat color finish and color finsh family as same
# (6222, 40733) 5211 treat color finish and color finsh family as same
test=pd.merge(df_train_proddesc_2,b,how='left',on='product_uid')
test.name_y.value_counts()


# In[ ]:



