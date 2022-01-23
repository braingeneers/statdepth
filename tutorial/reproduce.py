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
from statdepth.testing import generate_noisy_multivariate
from statdepth import PointcloudDepth
from statdepth.testing import generate_noisy_pointcloud 


# We'll now generate some random univariate functions with similar shape and some noise.

# In[2]:


df = generate_noisy_univariate(data=[2,3,3.4,4,5,3.1,3,3,2]*3, columns=[f'f{i}' for i in range(20)], seed=42)
print(df.head())

bd = FunctionalDepth([df], J=2, relax=False, quiet=False)

print(bd.deepest(n=5))
print(bd.outlying(n=5))

n=3
fig = bd.plot_deepest(n=n, return_plot=True, title=f'{n} Deepest Curves, Plotted in Red')
fig.update_layout(width=750, height=750)
fig.write_image('ex1_deepest.pdf')

fig = bd.plot_outlying(n=n, return_plot=True, title=f'{n} Most Outlying Curves in Red')
fig.update_layout(width=750, height=750)
fig.write_image('ex1_outlying.pdf')

print(bd.get_deepest_data(n=10).head())

data = generate_noisy_multivariate(columns=list('ABC'), num_curves=10, seed=42)
bd = FunctionalDepth(data, containment='simplex', relax=True, quiet=False)

print(bd.ordered())

df = generate_noisy_pointcloud(n=50, d=2, seed=42)
bd = PointcloudDepth(df, K=7, containment='simplex', quiet=False)

print(df.head())
print(bd.deepest(n=5))

fig = bd.plot_depths(invert_colors=True, return_plot=True, title='Pointcloud Depths, Deepest are Darkest')
fig.update_layout(width=750, height=750)
fig.write_image('ex2_colored.pdf')

fig = bd.plot_deepest(n=5, return_plot=True, title='5 Deepest Points Plotted in Red')
fig.update_layout(width=750, height=750)
fig.write_image('ex2_deepest.pdf')

n=10
fig = bd.plot_outlying(n=n, title=f'{n} Deepest Points Plotted in Red', return_plot=True)
fig.update_layout(width=750, height=750)
fig.write_image('ex2_outlying.pdf')


df = generate_noisy_pointcloud(n=75, d=3, columns=list('ABC'), seed=42)
bd = PointcloudDepth(df, K=10, containment='simplex')
print(df.head())

print(bd.deepest(n=5))

fig = bd.plot_depths(invert_colors=True, return_plot=True, title='Pointcloud Depths, Deepest are Darkest')
fig.update_layout(width=750, height=750)
fig.write_image('ex3_colored.pdf')

n=5
fig = bd.plot_deepest(n=n, title=f'{n} Deepest Points Plotted in Red', return_plot=True)
fig.update_layout(width=750, height=750)
fig.write_image('ex3_deepest.pdf')

fig = bd.plot_outlying(n=n, title=f'{n} Outlying Points Plotted in Red', return_plot=True)
fig.update_layout(width=750, height=750)
fig.write_image('ex3_outlying.pdf')

print(bd.deepest(n=3))
print(bd.get_deepest_data(n=5))

bd = PointcloudDepth(df, K=3, containment='l1')

fig = bd.plot_deepest(n=n, title=f'{n} Deepest Points Plotted in Red, L1 Depth', return_plot=True)
fig.update_layout(width=750, height=750)

print(bd.get_deepest_data(n=20))
print(bd.drop_outlying_data(n=25))

df_a = pd.read_excel('rat-trial-a.xlsx', sheet_name='Body weight').drop(['Trial_ID', 'Animal_ID'], axis=1).T
df_b = pd.read_excel('rat-trial-b.xlsx', sheet_name='Body weight').drop(['Trial_ID', 'Animal_ID'], axis=1).T
a_des = pd.read_excel('rat-trial-a.xlsx', sheet_name='Study design', header=1)
b_des = pd.read_excel('rat-trial-b.xlsx', sheet_name='Study design', header=1)

a_depths = FunctionalDepth([df_a], quiet=False, relax=True, K=10)
b_depths = FunctionalDepth([df_b], quiet=False, relax=True, K=10)


n=3
fig = a_depths.plot_deepest(n=n, title=f'{n} Deepest Rat Curves, Group A', return_plot=True, yaxis_title='Weight (g)')
fig.update_layout(width=750, height=750)
fig.write_image('rat_a_deepest.pdf')

fig = a_depths.plot_outlying(n=n, title=f'{n} Outlying Rat Curves, Group A', return_plot=True, yaxis_title='Weight (g)')
fig.update_layout(width=750, height=750)
fig.write_image('rat_a_outlying.pdf')

fig = b_depths.plot_deepest(n=n, title=f'{n} Deepest Rat Curves, Group B', return_plot=True, yaxis_title='Weight (g)')
fig.update_layout(width=750, height=750)
fig.write_image('rat_b_deepest.pdf')

fig = b_depths.plot_outlying(n=n, title=f'{n} Outlying Rat Curves, Group B', return_plot=True, yaxis_title='Weight (g)')
fig.update_layout(width=750, height=750)
fig.write_image('rat_b_outlying.pdf')


from statdepth.homogeneity import FunctionalHomogeneity

hom = FunctionalHomogeneity([df_a], [df_b], K=10, J=2, relax=True, method='p2')

df_a = pd.read_excel('rat-trial-a.xlsx', sheet_name='Body weight').drop(['Trial_ID'], axis=1)
df_b = pd.read_excel('rat-trial-b.xlsx', sheet_name='Body weight').drop(['Trial_ID'], axis=1)
a_des = pd.read_excel('rat-trial-a.xlsx', sheet_name='Study design', header=1)
b_des = pd.read_excel('rat-trial-b.xlsx', sheet_name='Study design', header=1)


data = pd.concat([df_a, df_b])
ann = pd.concat([a_des, b_des])
df_ann = data.merge(ann)

def gen_group(data, i):
    return data[data['Group_ID'] == i].drop(['Trial_ID', 'Cage_ID', 'Group_ID', 'Sex', 'Animal_ID'], axis=1).T

df1 = gen_group(df_ann, 1)
df2 = gen_group(df_ann, 2)
df3 = gen_group(df_ann, 3)
df4 = gen_group(df_ann, 4)

groups = [df1, df2, df3, df4]

n=3
for i, g in enumerate(groups):
    fig = FunctionalDepth([g], K=2, quiet=False, relax=True).plot_deepest(n, return_plot=True, title=f'{n} Deepest Curves, Experimental Group {i+1}')
    fig.update_layout(width=750, height=750)
    fig.write_image(f'rat_group_{i+1}.pdf')

from statdepth.homogeneity import FunctionalHomogeneity 

homs = np.zeros(shape=[4,4])

for i, d1 in enumerate(groups):
    for j, d2 in enumerate(groups):
        if i == j:
            homs[i, j] = 1
        else:
            hom = FunctionalHomogeneity([d1], [d2], K=10, J=2, relax=True, quiet=False)
            
            homs[i, j] = hom.homogeneity().values[0]
groups = [f'Group {i}' for i in range(1, 5)]


# just to remove annoying bug
import plotly.express as px 
fig = px.line([1,2,3], [1,2,3])
fig.write_image('IGNORE.pdf')

import plotly.graph_objects as go

fig = go.Figure(
    data=go.Heatmap(x=groups, y=groups, z=1-homs, colorscale='blues'),
    layout=go.Layout(title='P2 Homogeneity Coefficient for each Experimental Group')
)

fig.write_image('rat_homogeneity.pdf')