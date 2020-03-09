---
title: "NoMoney - Introduction"
date: 2020-03-01
tags: [nomoney]
categories: ds

excerpt: "How I learned to not make money"
usemathjax: "true"
---

# 3money0kids

Interested in making some money? Heard about those interesting things called "machine learning" or "cryptocurrency" recently? Trying to make use of your knowledge of math or programming to fatten up your wallet? Me too.  

I've spent the past several months reading and coding to help make myself some money with automated trading so that I wouldn't have to go and get a real job. Unfortunately, I'm sort of a dumbass and failed to make myself a millionaire over a few months; and I don't even know if the tiny successes I had were due to me being smart or if they were due to me being lucky.  

Nonetheless, I probably did learn a few things along the way that could be useful either to my future self or to others. So this is what this repo is for: to document some tools and techniques that can be used for something more than just slightly-informed gambling.

## Setting Up

I recommended copying [this repository](https://github.com/andy-vh/3money0kids) and using Anaconda to set up a virtual environment similar to mine in order to run these notebooks.

1. With Anaconda installed, use your terminal to create a virtual environment from the env.yml file provided

    conda env create -f env.yml
    
2. Once the environment is created, it can be easily activated with Anaconda

    conda activate nomoney
    
3. In order to use this virtual environment in Jupyter Notebook, you'll need to install the iPython kernel

    python -m ipykernel install --user --name nomoney
    
4. With everything set up, you can now activate Jupyter Notebook and explore the notebooks of your choice

    jupyter notebook
