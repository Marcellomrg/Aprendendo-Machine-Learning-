# %%
# Regressao logistica
import pandas as pd
from sklearn import linear_model
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
reg = linear_model.LogisticRegression()
reg
# %%
reg.fit(X,y)
# %%
predict = reg.predict(X[["cerveja"]].drop_duplicates())
predict_proba = reg.predict_proba(X[['cerveja']].drop_duplicates())[:,1]
print(predict_proba)
# %%
# Grafico Regressao Logistica

plt.plot(X,y,"o")
plt.xlabel("Cerveja")
plt.ylabel("Aprovacao")
plt.title("Cerveja vs Aprovacao")
plt.grid(True)
plt.plot(X['cerveja'].drop_duplicates(),predict,color="red")
plt.plot(X["cerveja"].drop_duplicates(),predict_proba)
plt.legend(["Obeservados","Logi Predict","Logi Proba"])
# %%
