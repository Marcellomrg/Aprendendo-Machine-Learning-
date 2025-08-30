# %%
# Utilizando o modelo de Naives Bayes para prever a aprovacao de um aluno com base no consumo de cerveja
import pandas as pd
from sklearn import naive_bayes as nb
import matplotlib.pyplot as plt

# %%
data = pd.read_excel("data\data/dados_cerveja_nota.xlsx")
data
# %%
data["Aprovados"] = (data["nota"] > 5).astype(int)
data
# %%
X = data[["cerveja"]]
y = data["Aprovados"]
# %%
bayes = nb.GaussianNB()
bayes
# %%
bayes.fit(X,y)
# %%
predict = bayes.predict(X.drop_duplicates()[["cerveja"]])
predict
# %%
proba = bayes.predict_proba(X.drop_duplicates()[["cerveja"]])[:,1]
proba

# %%

plt.figure(dpi=400)
plt.plot(X["cerveja"],y,"o")
plt.xlabel("Cerveja")
plt.ylabel("Aprovacao")
plt.title("Cerveja vs Aprovacao")
plt.grid(True)
plt.plot(X.drop_duplicates()["cerveja"],predict)
plt.plot(X.drop_duplicates()["cerveja"],proba)
plt.legend(["Observados","Bayes predict","Bayes proba"])


# %%
