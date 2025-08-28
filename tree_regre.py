# %%
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
# %%
data = pd.read_excel("data\data/dados_cerveja_nota.xlsx")
data
# %%
arvore = tree.DecisionTreeRegressor(random_state=42)
arvored2 = tree.DecisionTreeRegressor(random_state=42,max_depth=2)
# %%
X = data[['cerveja']]
y = data['nota']
print(X)
print(y)
# %%
arvore.fit(X,y)
arvored2.fit(X,y)
# %%
predict = arvore.predict(X.drop_duplicates())
predictd2 = arvored2.predict(X.drop_duplicates())
# %%
plt.figure(dpi=400)
plt.plot(X["cerveja"],y,"o")
plt.plot(X.drop_duplicates()["cerveja"],predict,color = 'green')
plt.grid(True)
plt.title("Nota vs Cerveja")
plt.xlabel("Cerveja")
plt.ylabel("Nota")
plt.plot(X.drop_duplicates()["cerveja"],predictd2,color = "magenta")
plt.legend(["Observados","Arvore full","max depth = 2"])
# %%
# %%
