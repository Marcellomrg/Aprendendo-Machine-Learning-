# %%
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("data\data/Dados_comunidade.csv")
df
# %%
df = df.replace({"Sim":1,"Não":0})

num_var = ["Curte games?","Curte futebol?"
           ,"Curte livros?","Curte jogos de tabuleiro?"
           ,"Curte jogos de fórmula 1?","Curte jogos de MMA?"
           ,"Idade"]



# %%
dummy_var = ["Como conheceu o Téo Me Why?"
               ,"Quantos cursos acompanhou do Téo Me Why?"
               ,"Estado que mora atualmente","Área de Formação"
               ,"Tempo que atua na área de dados"
               ,"Posição da cadeira (senioridade)"]


# %%
df_analise = pd.get_dummies(df[dummy_var]).astype(int)
df_analise
# %%
df_analise[num_var] = df[num_var].copy()
df_analise
# %%
df_analise["pessoa feliz"] = df["Você se considera uma pessoa feliz?"].copy()
df_analise
# %%

arvore = tree.DecisionTreeClassifier(random_state=42,min_samples_leaf=5)
arvore
# %%
features  = df_analise.columns[:-1].to_list()
features
# %%
x = df_analise[features]
y = df_analise["pessoa feliz"]

# %%
arvore.fit(x,y)
# %%
arvore_predict = arvore.predict(x)
arvore_predict
# %%
df_predict = df_analise[["pessoa feliz"]]
df_predict
# %%
df_predict["predict arvore"] = arvore_predict
df_predict
# %%
df_predict["predict proba"] = arvore.predict_proba(x)[:,1]
df_predict

# %%
# Acuracia
acc = (df_predict["pessoa feliz"] == df_predict["predict arvore"]).mean()
acc
# %%
pd.crosstab(df_predict['pessoa feliz'],df_predict['predict arvore'])
# %%
from sklearn import metrics

acc_arvore = metrics.accuracy_score(df_predict["pessoa feliz"],df_predict["predict arvore"])
acc_arvore
# %%
precisao_arvore = metrics.precision_score(df_predict["pessoa feliz"],df_predict["predict arvore"])
precisao_arvore
# %%
recall_arvore = metrics.recall_score(df_predict["pessoa feliz"],df_predict["predict arvore"])
recall_arvore

# %%
roc = metrics.roc_curve(df_predict["pessoa feliz"],df_predict["predict proba"])
roc
auc = metrics.roc_auc_score(df_predict["pessoa feliz"],df_predict["predict proba"])
# %%
plt.plot(roc[0],roc[1],"o-")
plt.grid(True)
plt.title("Curva ROC")
plt.xlabel("1 - Especificidade")
plt.ylabel("Recall")
plt.legend([f"Arvore: {auc:.2f}"])
# %%
