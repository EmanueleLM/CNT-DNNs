# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:18:12 2020

@author: Emanuele
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from colour import Color


def draw_bipartite_graph(ls, actual_size, title='', showfig=True, savefig=''):
    green = Color("green")
    colors = list(green.range_to(Color("red"),38))
    colors = [c.get_rgb() for c in colors]
    color_hists = np.histogram(ls.flatten(), 36)[1]
    g = nx.Graph()
    a=['a'+str(i) for i in range(actual_size[0])]
    b=['b'+str(j) for j in range(actual_size[1])]
    g.add_nodes_from(a,bipartite=0)
    g.add_nodes_from(b,bipartite=1)
    edges_color = []
    for i in range(actual_size[0]):
        for j in range(actual_size[1]):
            if ls[i][j] != 0.:
                g.add_edge(a[i], b[j])         
                edges_color.append(colors[np.digitize(ls[i,j],color_hists)])
                
    pos_a={}
    x=0.100
    const=0.100
    const_out = 0.300 
    y=1.0
    for i in range(len(a)):
        pos_a[a[i]]=[x,y-i*const]    
    xb=0.500
    pos_b={}
    for i in range(len(b)):
        pos_b[b[i]]=[xb,y-i*const_out]    
    nx.draw_networkx_nodes(g,pos_a,nodelist=a,node_color='r',node_size=30,alpha=0.8)
    nx.draw_networkx_nodes(g,pos_b,nodelist=b,node_color='b',node_size=30,alpha=0.8)    
    # edges
    pos={}
    pos.update(pos_a)
    pos.update(pos_b)
    # draw graph with colors for each link
    nx.draw_networkx_labels(g,pos,font_size=10,font_family='sans-serif')
    nx.draw_networkx_edges(g,pos,edgelist=nx.edges(g), edge_color=edges_color, width=2, alpha=0.8)    
    plt.title(title)
    if savefig != '':
        plt.savefig(savefig+'.png', bbox_inches='tight')
        #plt.savefig(savefig+'.svg', bbox_inches='tight')
    if showfig is True:
        plt.show()
    return g

    
