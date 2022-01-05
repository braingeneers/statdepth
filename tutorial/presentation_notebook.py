#!/usr/bin/env python
# coding: utf-8

# # statdepth: An Interactive Guide
# 
# In this notebook we'll be exploring `statdepth,` a Python package for computing statistical depth of univariate functional data, multivariate functional data, and pointcloud data for distributions in $\mathbb{R}^d$
# 
# We'll begin by importing some libraries we may need

# In[1]:


import numpy as np
import pandas as pd
from string import ascii_lowercase

from statdepth import FunctionalDepth, PointcloudDepth
from statdepth.testing import generate_noisy_pointcloud, generate_noisy_univariate


# We'll now generate some random univariate functions with similar shape and some noise.

# In[2]:


df = generate_noisy_univariate(data=[2,3,3.4,4,5,3.1,3,3,2]*3, columns=[f'f{i}' for i in range(20)], seed=42)
df.head()


# Now we'll use our library to calculate band depth (using standard containment on $\mathbb{R}^2$

# In[3]:


bd = FunctionalDepth([df], J=2, relax=False, quiet=False)


# Well, we can first look at the $n$ deepest and most outlying curves

# In[4]:


bd.deepest(n=5)


# In[5]:


bd.outlying(n=5)


# But this is much more meaningful with visuals!

# In[6]:


n=3
fig = bd.plot_deepest(n=n, return_plot=True, title=f'{n} Deepest Curves, Plotted in Red')
fig.update_layout(width=750, height=750)
fig.write_image('ex1_deepest.pdf')
fig.show()


# In addition to writing out the image in any general image format, we can visualize the results with Plotlys `.show()` method.

# We can also plot the most outlying functions

# In[7]:


n=3
fig = bd.plot_outlying(n=n, return_plot=True, title=f'{n} Most Outlying Curves in Red')
fig.update_layout(width=750, height=750)
fig.write_image('ex1_outlying.pdf')
fig.show()


# Or, supposing we've tuned our FunctionalDepth to our liking, return our data with the `n` most outlying samples dropped

# In[8]:


bd.get_deepest_data(n=10).head()


# We can calculate depth for multivariate data using simplex depth, which generalizes the idea of containment in 2 dimensions to functions $f: D \rightarrow \mathbb{R}^n$, where $D$ is a set of discrete time indices.

# In[9]:


from statdepth.testing import generate_noisy_multivariate

data = generate_noisy_multivariate(columns=list('ABC'), num_curves=10, seed=42)


# In[10]:


data[2]


# In[11]:


bd = FunctionalDepth(data, containment='simplex', relax=True, quiet=False)


# Again, we can look at our curves ordered

# In[12]:


bd.ordered()


# Now, let's try calculating band depth for some pointcloud data. Maybe you've sampled $n$ points from some distribution in $R^d$, and you'd like to understand which points are the most "central".
# 
# First, let's try this for some points sampled in $\mathbb{R^2}$

# In[13]:


from statdepth import PointcloudDepth
from statdepth.testing import generate_noisy_pointcloud 

df = generate_noisy_pointcloud(n=50, d=2, seed=42)
bd = PointcloudDepth(df, K=7, containment='simplex', quiet=False)
df.head()


# We can look at deepest points

# In[14]:


bd.deepest(n=5)


# Again, we can plot our data. Here, the lighter the color the deeper (more central) the point.

# In[15]:


fig = bd.plot_depths(invert_colors=True, return_plot=True, title='Pointcloud Depths, Deepest are Darkest')
fig.update_layout(width=750, height=750)
fig.write_image('ex2_colored.pdf')
fig.show()


# We can also just plot the $n$ deepest points. 

# In[16]:


fig = bd.plot_deepest(n=5, return_plot=True, title='5 Deepest Points Plotted in Red')
fig.update_layout(width=750, height=750)
fig.write_image('ex2_deepest.pdf')
fig.show()


# Or even the $n$ most outlying points, since often it's nice to know which data we should consider to be outliers

# In[17]:


n=10
fig = bd.plot_outlying(n=n, title=f'{n} Deepest Points Plotted in Red', return_plot=True)
fig.update_layout(width=750, height=750)
fig.write_image('ex2_outlying.pdf')
fig.show()


# But of course, if we're just defining depth using a certain measure of containment, there is no reason it shouldn't generalize to arbitrary dimensions. And indeed, this is the case. Let's take a look at some data in $\mathbb{R}^3$.
# 
# Notice, we're using sample depth because if we were to compute depth precisely, we'd be calculating about 500k simplices for each of our 50 datapoints, which can become unweildy fast.
# 
# However, it turns out that sample band depth is quite accurate for $K << n$, where $n$ is our number of datapoints, so this is definitely worth it.

# In[18]:


df = generate_noisy_pointcloud(n=75, d=3, columns=list('ABC'), seed=42)
bd = PointcloudDepth(df, K=10, containment='simplex')
df.head()


# In[19]:


bd.deepest(n=5)


# Well, looking at the 5 deepest points is interesting, but it's a lot more meaningful visually.

# In[20]:


fig = bd.plot_depths(invert_colors=True, return_plot=True, title='Pointcloud Depths, Deepest are Darkest')
fig.update_layout(width=750, height=750)
fig.write_image('ex3_colored.pdf')
fig.show()


# Or, we could just plot the $n$ deepest points

# In[21]:


n=5
fig = bd.plot_deepest(n=n, title=f'{n} Deepest Points Plotted in Red', return_plot=True)
fig.update_layout(width=750, height=750)
fig.write_image('ex3_deepest.pdf')
fig.show()

fig = bd.plot_outlying(n=n, title=f'{n} Outlying Points Plotted in Red', return_plot=True)
fig.update_layout(width=750, height=750)
fig.write_image('ex3_outlying.pdf')
fig.show()


# In[22]:


bd.deepest(n=3)


# In[23]:


bd.get_deepest_data(n=5)


# The above uses simplex containment, where to find the depth of a point in $\mathbb{R}^2$ we use all possible subsequences of 3 other points, construct a triangle, and check the proportion of triangles that our point is contained in.
# 
# We then to this for all other points we'd like to calculate depth for.
# 
# But this is not the only definition of depth. So let's use another below and see how it compares

# In[24]:


bd = PointcloudDepth(df, K=3, containment='l1')


# In[25]:


fig = bd.plot_deepest(n=n, title=f'{n} Deepest Points Plotted in Red, L1 Depth', return_plot=True)
fig.update_layout(width=750, height=750)
fig.show()


# Notice that this gives different deepest points than simplex depth. 

# In[26]:


bd.get_deepest_data(n=20)


# In[27]:


bd.drop_outlying_data(n=25)


# ## Rat data example

# In[2]:


import pandas as pd 
df_a = pd.read_excel('rat-trial-a.xlsx', sheet_name='Body weight').drop(['Trial_ID', 'Animal_ID'], axis=1).T
df_b = pd.read_excel('rat-trial-b.xlsx', sheet_name='Body weight').drop(['Trial_ID', 'Animal_ID'], axis=1).T
a_des = pd.read_excel('rat-trial-a.xlsx', sheet_name='Study design', header=1)
b_des = pd.read_excel('rat-trial-b.xlsx', sheet_name='Study design', header=1)
b_des


# In[3]:


a_depths = FunctionalDepth([df_a], quiet=False, relax=True, K=10)
b_depths = FunctionalDepth([df_b], quiet=False, relax=True, K=10)


# Now that we've calculated the depths for the rat curves, we can visualize the results, plotting the 3 deepest curves in red.

# In[7]:


n=3
fig = a_depths.plot_deepest(n=n, title=f'{n} Deepest Rat Curves, Group A', return_plot=True, yaxis_title='Weight (g)')
fig.update_layout(width=750, height=750)
fig.write_image('rat_a_deepest.pdf')
fig.show()

fig = a_depths.plot_outlying(n=n, title=f'{n} Outlying Rat Curves, Group A', return_plot=True, yaxis_title='Weight (g)')
fig.update_layout(width=750, height=750)
fig.write_image('rat_a_outlying.pdf')

fig.show()


# Similarly, we can do the same for experimental group B

# In[8]:


fig = b_depths.plot_deepest(n=n, title=f'{n} Deepest Rat Curves, Group B', return_plot=True, yaxis_title='Weight (g)')
fig.update_layout(width=750, height=750)
fig.write_image('rat_b_deepest.pdf')
fig.show()

fig = b_depths.plot_outlying(n=n, title=f'{n} Outlying Rat Curves, Group B', return_plot=True, yaxis_title='Weight (g)')
fig.update_layout(width=750, height=750)
fig.write_image('rat_b_outlying.pdf')

fig.show()


# However, a visual representation may not always be enough. For this reason, we can homogeneity test between the two trials, and check if they are distributionally similar, an assumption we hope holds true

# In[9]:


from statdepth.homogeneity import FunctionalHomogeneity

hom = FunctionalHomogeneity([df_a], [df_b], K=10, J=2, relax=True, method='p2')


# In[10]:


hom.homogeneity()[0]


# In[11]:


hom


# In[4]:


df_a = pd.read_excel('rat-trial-a.xlsx', sheet_name='Body weight').drop(['Trial_ID'], axis=1)
df_b = pd.read_excel('rat-trial-b.xlsx', sheet_name='Body weight').drop(['Trial_ID'], axis=1)
a_des = pd.read_excel('rat-trial-a.xlsx', sheet_name='Study design', header=1)
b_des = pd.read_excel('rat-trial-b.xlsx', sheet_name='Study design', header=1)


# In[5]:


data = pd.concat([df_a, df_b])


# In[6]:


ann = pd.concat([a_des, b_des])


# In[7]:


df_ann = data.merge(ann)


# In[8]:


df_ann


# In[9]:


def gen_group(data, i):
    return data[data['Group_ID'] == i].drop(['Trial_ID', 'Cage_ID', 'Group_ID', 'Sex', 'Animal_ID'], axis=1).T

df1 = gen_group(df_ann, 1)
df2 = gen_group(df_ann, 2)
df3 = gen_group(df_ann, 3)
df4 = gen_group(df_ann, 4)

groups = [df1, df2, df3, df4]


# In[18]:


n=3
for i, g in enumerate(groups):
    fig = FunctionalDepth([g], K=2, quiet=False, relax=True).plot_deepest(n, return_plot=True, title=f'{n} Deepest Curves, Experimental Group {i+1}')
    fig.update_layout(width=750, height=750)
    fig.write_image(f'rat_group_{i+1}.pdf')
    fig.show()


# Finally, we can calculate the homogeneity coefficients between each group, and visualize this as a heatmap. Note that the homogeneity coefficients are not directly symmetric, and this is expected.

# In[11]:


from statdepth.homogeneity import FunctionalHomogeneity 

homs = np.zeros(shape=[4,4])

for i, d1 in enumerate(groups):
    for j, d2 in enumerate(groups):
        if i == j:
            homs[i, j] = 1
        else:
            hom = FunctionalHomogeneity([d1], [d2], K=10, J=2, relax=True, quiet=False)
            
            homs[i, j] = hom.homogeneity().values[0]


# In[15]:





# In[14]:


groups = [f'Group {i}' for i in range(1, 5)]


# In[30]:


# just to remove annoying bug
import plotly.express as px 
fig = px.line([1,2,3], [1,2,3])
fig.write_image('IGNORE.pdf')


# In[32]:


import plotly.graph_objects as go

fig = go.Figure(
    data=go.Heatmap(x=groups, y=groups, z=1-homs, colorscale='blues'),
    layout=go.Layout(title='P2 Homogeneity Coefficient for each Experimental Group')
)

fig.show()


# In[33]:


fig.write_image('rat_homogeneity.pdf')


# In[ ]:




