# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
clones = pd.read_parquet("data\data/dados_clones.parquet")
clones.head(50)
# %%
arvore = tree.DecisionTreeClassifier(random_state=42,max_depth=3)

clones = clones.replace({"Tipo 4":4,"Tipo 3":3,"Tipo 2":2,"Tipo 1":1,"Tipo 5":5})
clones.head()
# %%
features = ["Massa(em kilos)","Estatura(cm)"
            ,"Distância Ombro a ombro"
            ,"Tamanho do crânio"
            ,"Tamanho dos pés","Tempo de existência(em meses)"]

y = clones["Status "]
x = clones[features]

arvore.fit(x,y)
# %%
plt.figure(dpi=400)
tree.plot_tree(arvore,feature_names=features,class_names=arvore.classes_,filled=True,max_depth=3)
# %%
