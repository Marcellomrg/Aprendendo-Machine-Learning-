# %%
# Usando a arvore de classificacao para prever os alunos aprovados com nota maior que 5
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

# %%
data = pd.read_excel("data\data/dados_cerveja_nota.xlsx")
data
# %%
data["Aprovado"] = (data["nota"] > 5).astype(int)
data
# %%
X = data[["cerveja"]]
X
# %%
y = data["Aprovado"]
y
# %%
arvore = tree.DecisionTreeClassifier()
arvore
# %%
arvore.fit(X,y)
# %%
predict = arvore.predict(X[["cerveja"]].drop_duplicates())
predict
# %%
predict_proba = arvore.predict_proba(X[["cerveja"]].drop_duplicates())[:,1]
predict_proba

# %%
plt.figure(dpi=400)
plt.plot(X["cerveja"],y,"o")
plt.grid(True)
plt.xlabel("ceverjas")
plt.ylabel("Aprovacao")
plt.title("Cervejas vs Aprovacao")
plt.plot(X.drop_duplicates()['cerveja'],predict)
plt.plot(X.drop_duplicates()['cerveja'],predict_proba)
plt.hlines(0.5,xmin=1,xmax=9,colors="black",linestyles="--")
plt.legend(["Observados","Arvore predict","Arvore proba"])
# %%
