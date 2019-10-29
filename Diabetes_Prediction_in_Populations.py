#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html

import cufflinks as cf
from plotly import graph_objs as go
import plotly.offline
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("./data/pima-data.csv")
df_copy = df.copy(deep=True)


# In[3]:


df.shape
df.head()


# # Cleaning up the data in the 'diabetes' column (making it numerical)

# In[4]:


diabetes_map = {True:1, False:0}
df['diabetes'] = df['diabetes'].map(diabetes_map)
df.head()


# In[5]:


#df.corr().iplot(kind='heatmap', colorscale="Blues", title="Feature Correlation Matrix", fontsize=10)


# In[6]:


#plt.figure(figsize=(14,10))
#sns.heatmap(df.corr(),
#            vmin=-1,
 #           cmap='coolwarm',
 #           annot=True);


# ## We have a 1:1 correlation between skin and thickness so we will drop the skin feature

# In[7]:


df = df.drop('skin', 1)


# In[8]:


#df.corr().iplot(kind='heatmap', colorscale="Blues", title="Feature Correlation Matrix")


# In[9]:


#plt.figure(figsize=(14,10))
#sns.heatmap(df.corr(),
#            vmin=-1,
#            cmap='coolwarm',
#            annot=True);


# In[10]:


print(df.diabetes.value_counts())

#p=df.diabetes.value_counts().plot(kind='bar')


# ## Check for invalid zero values
# The min column shows us we have zero values where their should not be

# In[11]:


df.describe().T


# ## Replace missing/zero values with NaN so we can later fix these values
# columns glucose_conc, diastolic_bp, thickness, insulin, and bmi cannot/should not have zero as a value.
# We will fix this now

# In[12]:


df[['glucose_conc', 
    'diastolic_bp', 
    'thickness', 
    'insulin', 
    'bmi']] = df[['glucose_conc',
                  'diastolic_bp', 
                  'thickness', 
                  'insulin', 
                  'bmi']].replace(0, np.NaN)


# In[13]:


print(df.isnull().sum())


# ## Imputing new values for the NaN values

# In[14]:


df['glucose_conc'].fillna(df['glucose_conc'].median(), inplace = True)
df['diastolic_bp'].fillna(df['diastolic_bp'].median(), inplace = True)
df['thickness'].fillna(df['thickness'].median(), inplace = True)
df['insulin'].fillna(df['insulin'].median(), inplace = True)
df['bmi'].fillna(df['bmi'].median(), inplace = True)


# In[15]:


#p = df.hist(figsize=(13,13), color='g')


# ## Preprocessing

# In[16]:


X = df.drop('diabetes', 1)
y = df['diabetes']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)


# ## Normalize our data

# In[17]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Applying PCA - our descriptive method

# In[18]:


from sklearn.decomposition import PCA

pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# ## Find the "explained variance ratio"

# In[19]:


pca.explained_variance_ratio_


# In[20]:


df_variance = pd.DataFrame({'var':pca.explained_variance_ratio_,
             'PC':['PC1','PC2','PC3','PC4', 'PC5', 'PC6', 'PC7', 'PC8']})
#sns.barplot(x='PC',y="var", 
#           data=df_variance, color="c");


# ## PC5-PC9 are not really contributing so we will eliminate them

# In[21]:


pca = PCA(n_components=4)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

pca.explained_variance_ratio_


# In[22]:


df_variance = pd.DataFrame({'var':pca.explained_variance_ratio_,
             'PC':['PC1','PC2','PC3','PC4']})
#sns.barplot(x='PC',y="var", 
#           data=df_variance, color="g");


# ## get_score() used to maintain the product
# Use the get_score method to easily check multiple alternate ML algorithms so we can 
# maintain the highest level of predictive accuracy in the future.

# In[23]:


## Add method to test multiple ML algs here
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)*100


# ## Training the model using Logistic Regression - non-descriptive method

# In[24]:


from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(solver='liblinear', penalty='l2',C = 0.001,random_state = 42)

LR.fit(X_train, y_train)

y_pred = LR.predict(X_test)


# ## Checking our Accuracy

# In[25]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print('Accuracy: ' , round(accuracy_score(y_test, y_pred)*100, 2),'%')


# ## Confirm using a Confusion Matrix

# In[26]:


sns.set(font_scale=1.5)
cm = confusion_matrix(y_test, y_pred)
#sns.heatmap(cm, annot=True, fmt='g')
#plt.show()


# # Create our App

# In[27]:


from dash.dependencies import Input, Output
import dash_auth

df_copy['d_positive']=df[df['diabetes']==1]['age']
df_copy['d_negative']=df[df['diabetes']==0]['age']


histogram = df_copy[['d_negative', 'd_positive']].iplot(kind='histogram', 
                                                        barmode='stack',
                                                        bins=20, 
                                                        theme='white', 
                                                        title='Diabetics per Age Group', 
                                                        xTitle='Ages', 
                                                        yTitle='Count', 
                                                        asFigure=True)

corr_matrix = df.corr().iplot(kind='heatmap', 
                              colorscale="Blues", 
                              title="Feature Correlation Matrix", 
                              theme='white', 
                              asFigure=True)

                                                                      
VALID_USERNAME_PASSWORD_PAIRS = [
    ['username', 'password']
]

   
    
external_stylesheets = ['https://codepen.io/amyoshino/pen/jzXypZ.css']
app = dash.Dash(external_stylesheets=external_stylesheets)
server = app.server

auth = dash_auth.BasicAuth(
    app, 
    VALID_USERNAME_PASSWORD_PAIRS
)


app.title = 'Diabetes Prediction'

app.layout = html.Div([
    
    #Header
    html.Div([
        html.H1('Diabetes Prediction in Populations'),
        html.Div('Predict the onset of diabetes based on diagnostic measures'),
        html.Div('Drag and select part of any graph to zoom in and inspect.  Double click graph to reset.')
    ], className = "row"),
    #Charts/Graphs
    html.Div([
        html.Div([
            
            dcc.Graph(id='stacked_bar_chart', figure = histogram)
        ], className = "seven columns"),
        
        html.Div([
            
            dcc.Graph(id='correlation_matrix', figure = corr_matrix)
        ], className = "five columns")
    ], className="row"),
    #Dummy row for formatting
    html.Div([
        html.H1('')
    ], className = 'row'),
    #Data tables Row
    html.Div([
        # Data Table
        html.Div([
           #html.H3('Patient Data'),
           html.H6('Select check boxes on the left to explore and compare patient data.  Sort by any column by clicking on the arrows by the column name.'),
           #html.Div(children='''
           #    Select patients on the left to explore compare data.
           #    Sort by any feature by clicking on the arrows by the feature name.
           #'''),
           dash_table.DataTable(
               id='Table',
               columns = [{'name': i, 'id': i} for i in df.columns],
               data=df.to_dict('records'),
               style_table={
                   'height': '300px',
                   'overflowY': 'scroll'
               },
               style_header={
                   'backgroundColor': 'rgb(230, 230, 230)',
                   'fontWeight': 'bold'
               },
               sort_action='native',
               row_selectable='multiple',
               selected_rows=[0],
               fixed_rows={'headers': True, 'data': 0},
               style_cell_conditional=[
                   {'if': {'column_id': 'num_preg'},'width': '60px'},
                   {'if': {'column_id': 'glucose_conc'}, 'width': '75px'},
                   {'if': {'column_id': 'diastolic_bp'}, 'width': '75px'},
                   {'if': {'column_id': 'thickness'},'width': '60px'},
                   {'if': {'column_id': 'insulin'}, 'width': '60px'},
                   {'if': {'column_id': 'bmi'}, 'width': '60px'},
                   {'if': {'column_id': 'diab_pred'},'width': '100px'},
                   {'if': {'column_id': 'age'}, 'width': '60px'},
                   {'if': {'column_id': 'diabetes'}, 'width': '60px'},
               ],
               
           )
        ], className = 'six columns'),
        
        #Linked Subject Bar Chart
        html.Div([
            dcc.Graph(id='linked_histogram'), 
        ], className = 'six columns'),
        
    ], className = 'row')
    
    
])


@app.callback(
    Output('linked_histogram', 'figure'),
    [Input('Table', 'selected_rows')])
def update_graph(sr):
    
    d = df.iloc[sr]
    d = d.reset_index(drop=True)
    
    return d[['num_preg', 
              'glucose_conc',
              'diastolic_bp', 'thickness', 
              'insulin', 
              'bmi', 
              'diab_pred',
              'age']].iplot(kind='bar', # alt kind='area' or kind='scatter' or kind='bar'
                            barmode='grouped', 
                            theme='white', 
                            title = 'Click feature names on the right to add or remove to customize comparisons.',
                            fill=True , 
                            asFigure=True)


# ## To sign into the app use the user name 'username' and the password 'password'

# In[ ]:


if __name__ == '__main__':
    app.run_server()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




