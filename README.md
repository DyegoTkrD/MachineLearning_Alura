# Projeto de Machine Learning - Classificação de Churn

Este é um projeto de Machine Learning para a classificação de churn utilizando Python. O objetivo é prever se um cliente vai ou não cancelar um serviço, baseando-se em um conjunto de dados fornecido.

## Dados

Os dados utilizados neste projeto foram carregados a partir da seguinte URL:

```python
URI = 'https://raw.githubusercontent.com/alura-cursos/ML_Classificacao_por_tras_dos_panos/main/Dados/Customer-Churn.csv'
dados = pd.read_csv(URI)
```
O conjunto de dados inclui informações sobre clientes e a variável alvo é o indicador de churn.

## Bibliotecas Utilizadas
1. Pandas
2. Seaborn
3. Sklearn

## Exploração dos Dados

```python
# Exibindo as primeiras linhas dos dados
dados.shape
dados.head()
```

Realiza-se uma modificação manual nos dados:

```python
# Modificação manual
traducao_dic = {
    'Sim': 1,
    'Nao': 0
}

dadosmodificados = dados[['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn']].replace(traducao_dic)
dadosmodificados.head()
```

Em seguida, são criadas variáveis dummy para as colunas categóricas:

```python
dumies_dados = pd.get_dummies(dados.drop(['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn'], axis=1))
dados_final = pd.concat([dadosmodificados, dumies_dados], axis=1)
dados_final.head()
dados_final.shape
```

## Balanceamento dos dados
```python
smt = SMOTE(random_state=123)  
X, y = smt.fit_resample(X, y)  
dados_final = pd.concat([X, y], axis=1)  
dados_final.head(2)
```

## Separação da base em teste e treino
```python
from sklearn.model_selection import train_test_split
X_treino, X_teste, y_treino, y_teste = train_test_split(X_normalizado, y, test_size=0.3, random_state=123)
```

## K-Nearest Neighbors (KNN)
```python
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(metric='euclidean')
knn.fit(X_treino, y_treino)
predito_knn = knn.predict(X_teste)
```
## Naive Bayes Bernoulli (BNB)
```python
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB(binarize=0.44)
bnb.fit(X_treino, y_treino)
predito_bnb = bnb.predict(X_teste)
```

## Decision Tree Classifier
```python
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy', random_state=42)
dtc.fit(X_treino, y_treino)
predito_dtc = dtc.predict(X_teste)
```

## Validando os modelos
```python
predito = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]
real = [1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0]
print("Recall: {:.2f}%".format(recall_score(real, predito)*100))
print("F1 Score: {:.2f}%".format(f1_score(real, predito)*100))
```

Sinta-se à vontade para contribuir, fornecer feedback ou utilizar este código como ponto de partida para seus próprios projetos de Machine Learning.
