# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
# %%
frutas = pd.read_excel("data\data/dados_frutas.xlsx")
frutas
# %%
arvore = tree.DecisionTreeClassifier(random_state=42)
arvore
# %%
y = frutas["Fruta"]
caracteristicas = ["Arredondada","Suculenta","Vermelha","Doce"]
x = frutas[caracteristicas]

# %%
# Machiche learning
arvore.fit(x,y)
# %%
arvore.predict([[1,0,0,0]])
# %%
plt.figure(dpi=400)
tree.plot_tree(arvore,feature_names=caracteristicas
               ,class_names=arvore.classes_,filled=True)



# %%
proba = arvore.predict_proba([[1,1,1,1]])[0]
proba
# %%
pd.Series(proba,index=arvore.classes_)

# %%

