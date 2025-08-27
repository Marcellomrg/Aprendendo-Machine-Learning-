# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
# %%
cerveja = pd.read_excel("data\data/dados_cerveja.xlsx")
cerveja
# %%
cerveja = cerveja.replace({"mud":1,"pint":2
                 ,"não":0,"sim":1
                 ,"escura":1,"clara":2})
cerveja


# %%
arvore = tree.DecisionTreeClassifier(random_state=42)
arvore
# %%
y = cerveja["classe"]
caracteristicas = ["temperatura","copo","espuma","cor"]
x= cerveja[caracteristicas]
arvore.fit(x,y)

# %%
plt.figure(dpi=400)
tree.plot_tree(arvore,feature_names=caracteristicas,class_names=arvore.classes_,filled=True)

# %%
