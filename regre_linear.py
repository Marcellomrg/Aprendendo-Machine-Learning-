# %%
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
# %%
dados = pd.read_excel("data\data/dados_cerveja_nota.xlsx")
dados
# %%
X = dados[['cerveja']]
y = dados['nota']
# %%
reg = linear_model.LinearRegression()
reg
# %%
reg.fit(X,y)

# %%
a = reg.intercept_
print(a)
b = reg.coef_[0]
print(b)
# %%
predict = reg.predict(X.drop_duplicates())
# %%
plt.plot(X['cerveja'],y,'o')
plt.grid(True)
plt.xlabel("Cerveja")
plt.ylabel("Notas")
plt.title("Notas vs cervejas")
plt.plot(X.drop_duplicates()['cerveja'],predict)
plt.legend(['Obeservado',f"y = {a:.3f} + {b:.3f} x"])
# %%

# %%
