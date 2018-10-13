
# coding: utf-8

# ENRON DATASET - Identificando uma fraude através dos emails e dados financeiros da Enron

#  A empresa Enron era uma empresa americana de energia, commodities e serviços com sede em Houston, Texas, que foi uma das maiores empresas dos Estados Unidos. Em 2002, a empresa entrou em colapso devido à grande quantidade de fraudes corporativas e contábeis. Seu colapso afetou milhares de funcionários e influenciou todo o sistema econômico ocidental.
#  
#  O objetivo deste projeto é usar dados financeiros e de e-mail dos executivos da empresa Enron, que foram liberados pelo governo dos EUA após a investigação efetuada, para chegar a um modelo preditivo que possa identificar pessoas possivelmente envolvidas na fraude. 
# 
# 

# In[1]:


#!/usr/bin/python3

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from IPython.display import Image
import matplotlib.pyplot as plt


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'to_messages', 'deferral_payments', 'total_payments', 
                 'loan_advances', 'bonus', 'restricted_stock_deferred', 
                 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 
                 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "rb") )


# In[2]:


print ('Numero de executivos no Dataset:', len(data_dict.keys()))


# In[3]:


poi = 0
for people in data_dict:
    if data_dict[people]['poi'] == 1:
        poi += 1
print ("Numero de pessoas de interesse (POI): ", poi)


# In[4]:


print (data_dict['BUY RICHARD B'])


# In[5]:


print (data_dict.values())


# Temos 146 registros no arquivo. Desse total, 18 são classificados como pessoas de interesse, o que dá entre 12% a 13% do dataset.
# 
# Os dados estão classificados como Pessoas de Interesse (POI), 14 recursos financeiros e 6 recursos de e-mail.

# #1 - Vamos procurar por Outliers

# In[6]:


features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")


# Ao analisar os valores de Salário e Bonus anual dos executivos, encontramos um valor extremamente discrepante. Vamos analisar os principais valores para ver do que se trata e entender melhor os dados.

# In[7]:


data_salary= []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    data_salary.append((key,int(val)))

print(sorted(data_salary,key=lambda x:x[1],reverse=True)[:5])


# O valor discrepante é o 'TOTAL'. Como não se trata de nenhum dos executivos, vou removê-lo e verificar novamente o dataset.

# In[8]:


data_dict.pop('TOTAL', 0)


# In[9]:



features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


# In[10]:


for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")



# Agora podemos ver alguns valores bem discrepantes, mas que foram realmente executivos da Enron. 

# In[11]:


data_salary= []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    data_salary.append((key,int(val)))

print(sorted(data_salary,key=lambda x:x[1],reverse=True)[:5])


# Vamos analisar nosso dicionario visualmente, buscando alguma outra informação incorreta além do 'TOTAL'. 

# In[12]:


data_dict.keys()


# Verificando os dados no dicionario, encontramos dois nomes suspeitos que devem ser analisados - THE TRAVEL AGENCY IN THE PARK e 'CHRISTODOULOU DIOMEDES'.
# O primeiro realmente deve ser cancelado, pois não se trata de uma pessoa.
# O segundo realmente é uma pessoa. Christodoulou é um nome de origem grega.
# Desse modo, vamos retirar THE TRAVEL AGENCY IN THE PARK

# In[13]:


data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)


# Vamos criar dois novos rotulos para analisar os dados.
# Como visto em classe e por intuição, vamos usar a fração da quantidade de mensagens enviadas para POI's e a fração de mensagens recebidas de POI's de cada executivo.
# 
# 

# In[14]:



def novo_dict(key,normalizer):
    nova_lista=[]
    for f in data_dict:
        if data_dict[f][key]=="NaN" or data_dict[f][normalizer]=="NaN":
            nova_lista.append(0.)
        elif data_dict[f][key]>=0:
            nova_lista.append(float(data_dict[f][key])/float(data_dict[f][normalizer]))
    return nova_lista

### create two lists of new features
from_poi_fraction =novo_dict("from_poi_to_this_person","to_messages")
to_poi_fraction =novo_dict("from_this_person_to_poi","from_messages")

count=0
for i in data_dict:
    data_dict[i]["from_poi_fraction"]=from_poi_fraction[count]
    data_dict[i]["to_poi_fraction"]=to_poi_fraction[count]
    count +=1


# In[15]:



features_list = ["poi", "from_poi_fraction", "to_poi_fraction"]    
### store to my_dataset for easy export below
my_dataset = data_dict


### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(data_dict, features_list)

### plot new features


# In[16]:



for point in data:
    from_poi = point[1]
    to_poi = point[2]
    plt.scatter( from_poi, to_poi )
    if point[0] == 1:
        plt.scatter(from_poi, to_poi, color="r", marker="*")
plt.xlabel("fraction of emails this person gets from poi")
plt.show()

    


# Como pode ser visto no grafico acima, estes rotulos não são muito eficientes sozinhos.
# Vamos usá-los posteriormente em nossos classificadores para verificar seu valor em um algoritmo preditivo.

# Vamos carregar a features_list novamente com todos os valores e guardar o dataset para facil exportação posterior.

# In[17]:


features_list = ['poi','salary', 'to_messages', 'deferral_payments', 'total_payments', 
                 'loan_advances', 'bonus', 'restricted_stock_deferred', 
                 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 
                 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees', 'from_poi_fraction', 'to_poi_fraction']


# In[18]:



my_dataset = data_dict
data = featureFormat(data_dict, features_list)


# Vamos utilizar os classificadores KbestNeighbors, Decision Tree, Gaussian NB e SVM.
# inicialmente irei testar os classificadores com todas as features, apenas para fins de comparação.

# In[19]:


from sklearn import cross_validation

labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.33, random_state=42)


# In[20]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Estudando novamente as métricas de precision e recall e o classificador SVM, verifiquei que o parametro Average é necessário para destinos multiclasse. Neste caso, classificando como "average = 'macro'", as métricas para cada label são calculadas, e encontramos sua média não ponderada. Isso não leva em conta o desequilíbrio do label.

# In[21]:



clf = GaussianNB()
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
acuracia = accuracy_score(labels_test, predictions)
precisao = precision_score(labels_test, predictions)
recall = recall_score(labels_test, predictions)


print ("acuracia:", acuracia)
print ("Precisão:", precisao)
print ("Recall:", recall)


# In[22]:


clf = KNeighborsClassifier(5)
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
acuracia = accuracy_score(labels_test, predictions)
precisao = precision_score(labels_test, predictions)
recall = recall_score(labels_test, predictions)
print ("accuracy: ",acuracia)
print ("precision: ",precisao)
print ("recall: ",recall)


# In[23]:


clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
acuracia = accuracy_score(labels_test, predictions)
precisao = precision_score(labels_test, predictions)
recall = recall_score(labels_test, predictions)
print ("accuracy: ",acuracia)
print ("precision: ",precisao)
print ("recall: ",recall)


# In[24]:


clf = SVC()
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
acuracia = accuracy_score(labels_test, predictions)
precisao = precision_score(labels_test, predictions, average = 'macro')
recall = recall_score(labels_test, predictions, average = 'macro')
print ("accuracy: ",acuracia)
print ("precision: ",precisao)
print ("recall: ",recall)


# Obtivemos excelentes resultados de acurácia com SVM, KNeighbors e Decision Tree,mas resultados ruins para Precisão e Recall, além de SVM ter recall zero
# Vamos utilizar o algoritimo Select Kbest para determinar as melhores Features para otimizar as predições e obter melhores resultados de Acuracia, Precisão e Recall.

# In[27]:


def  get_k_best (data_dict, features_list, k):

    data_kbest = featureFormat(data_dict, features_list)
    k_labels, k_features = targetFeatureSplit(data_kbest)

    k_best = SelectKBest(k=5)
    k_best.fit(k_features, k_labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print ("{0} best features: {1}\n".format(k, k_best_features.keys()))
    print (k_best_features)
    return (k_best_features)


# In[28]:



target_label = 'poi'
from sklearn.feature_selection import SelectKBest
num_features = 15
best_features = get_k_best(data_dict, features_list, num_features)


# Selecionei as 15 melhores Features para ter uma melhor noção de quais poderiam ser usadas.
# Vamos testar os classificadores com as 4 melhores.
# 

# In[29]:


my_dataset = data_dict
features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary']
data = featureFormat(my_dataset, features_list)



# In[30]:


labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.33, random_state=42)


# In[31]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
acuracia = accuracy_score(labels_test, predictions)
precisao = precision_score(labels_test, predictions)
recall = recall_score(labels_test, predictions)

print ("acuracia:", acuracia)
print ("Precisão:", precisao)
print ("Recall:", recall)


# In[32]:


clf = KNeighborsClassifier(3)
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
acuracia = accuracy_score(labels_test, predictions)
precisao = precision_score(labels_test, predictions)
recall = recall_score(labels_test, predictions)
print ("acuracia:", acuracia)
print ("Precisão:", precisao)
print ("Recall:", recall)


# In[33]:


clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
acuracia = accuracy_score(labels_test, predictions)
precisao = precision_score(labels_test, predictions)
recall = recall_score(labels_test, predictions)
print ("acuracia:", acuracia)
print ("Precisão:", precisao)
print ("Recall:", recall)


# In[34]:


clf = SVC()
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
acuracia = accuracy_score(labels_test, predictions)
precisao = precision_score(labels_test, predictions, average = 'macro')
recall = recall_score(labels_test, predictions, average = 'macro')
print ("acuracia:", acuracia)
print ("Precisão:", precisao)
print ("Recall:", recall)


# Os valores de precisao e recall ficaram baixos para todos os classificadores.
# 
# Usando a intuição, vamos usar outras features e ver se alguma melhora.

# In[37]:


features_list = [ 'poi','salary', 'bonus', 'from_poi_fraction', 'to_poi_fraction']

data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.33, random_state=42)


# In[38]:



clf = GaussianNB()
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
acuracia = accuracy_score(labels_test, predictions)
precisao = precision_score(labels_test, predictions)
recall = recall_score(labels_test, predictions)
print ("acuracia:", acuracia)
print ("Precisão:", precisao)
print ("Recall:", recall)


# In[39]:


clf = KNeighborsClassifier(5)
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
acuracia = accuracy_score(labels_test, predictions)
precisao = precision_score(labels_test, predictions)
recall = recall_score(labels_test, predictions)
print ("acuracia:", acuracia)
print ("Precisão:", precisao)
print ("Recall:", recall)


# In[40]:


clf = DecisionTreeClassifier(max_depth=3)
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
acuracia = accuracy_score(labels_test, predictions)
precisao = precision_score(labels_test, predictions)
recall = recall_score(labels_test, predictions)
print ("acuracia:", acuracia)
print ("Precisão:", precisao)
print ("Recall:", recall)


# In[42]:


clf = SVC()
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
acuracia = accuracy_score(labels_test, predictions)
precisao = precision_score(labels_test, predictions, average = 'macro')
recall = recall_score(labels_test, predictions, average = 'macro')
print ("acuracia:", acuracia)
print ("Precisão:", precisao)
print ("Recall:", recall)


# Houve uma ligeira melhora na acuracia, mas uma queda no Recall.
# Vamos melhorar os parametros dos classificadores SVM e Decision Tree, usando com GridSearchCV, afim de afinar os algoritmos.
# 

# In[45]:


from sklearn.model_selection import GridSearchCV



parameters = {}
clf_teste = SVC()
clf_teste = GridSearchCV(clf_teste, parameters)
clf_teste.fit(features, labels)

print (clf_teste.best_params_)
print (clf_teste.best_estimator_)


# In[47]:


parameters = {}
clf_teste = clf = DecisionTreeClassifier()
clf_teste = GridSearchCV(clf_teste, parameters)
clf_teste.fit(features, labels)

print (clf_teste.best_params_)
print (clf_teste.best_estimator_)


# In[48]:


clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
acuracia = accuracy_score(labels_test, predictions)
precisao = precision_score(labels_test, predictions, average = 'macro')
recall = recall_score(labels_test, predictions, average = 'macro')
print ("acuracia:", acuracia)
print ("Precisão:", precisao)
print ("Recall:", recall)


# In[49]:


clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
acuracia = accuracy_score(labels_test, predictions)
precisao = precision_score(labels_test, predictions)
recall = recall_score(labels_test, predictions)
print ("acuracia:", acuracia)
print ("Precisão:", precisao)
print ("Recall:", recall)


# Tivemos uma boa melhora no Recall.
# 
# Houve um erro ao executar o SVM, informando que o máximo de iterações foi 100, e o estabelecido era 1000.
# Sugeriram o uso dos algoritimos StandarScaler e MinMaxScaler.
# Deste modo, vamos fazer o escalonamento das features implementando o algoritmo MinMaxScaler e testar os classificadores novamente.

# In[51]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features= scaler.fit_transform(features)


# In[86]:



clf = GaussianNB()
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
acuracia = accuracy_score(labels_test, predictions)
precisao = precision_score(labels_test, predictions)
recall = recall_score(labels_test, predictions)
print ("acuracia:", acuracia)
print ("Precisão:", precisao)
print ("Recall:", recall)


# In[78]:


clf = KNeighborsClassifier(leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=5,
           weights='uniform')
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
acuracia = accuracy_score(labels_test, predictions)
precisao = precision_score(labels_test, predictions)
recall = recall_score(labels_test, predictions)
print ("acuracia:", acuracia)
print ("Precisão:", precisao)
print ("Recall:", recall)


# In[83]:


clf = DecisionTreeClassifier(max_depth=5)
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
acuracia = accuracy_score(labels_test, predictions)
precisao = precision_score(labels_test, predictions)
recall = recall_score(labels_test, predictions)
print ("acuracia:", acuracia)
print ("Precisão:", precisao)
print ("Recall:", recall)


# In[82]:


clf = SVC()
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
acuracia = accuracy_score(labels_test, predictions)
precisao = precision_score(labels_test, predictions, average = 'macro')
recall = recall_score(labels_test, predictions, average = 'macro')
print ("acuracia:", acuracia)
print ("Precisão:", precisao)
print ("Recall:", recall)


# Todos os algoritmos apresentaram uma melhora significativa, sendo o melhor no momento o Kneighbors.
# Como já usei o GridSearchCV, vou ajustar os parametros dos algoritmos DecisionTree e SVM manualmente, e decidir pelo classificador final.
# 

# In[115]:


clf = DecisionTreeClassifier(criterion = 'entropy', min_samples_split= 6, max_depth=5)
clf.fit(features_train, labels_train)
clf.get_params

predictions = clf.predict(features_test)
acuracia = accuracy_score(labels_test, predictions)
precisao = precision_score(labels_test, predictions)
recall = recall_score(labels_test, predictions)
print ("Acuracia:", acuracia)
print ("Precisão:", precisao)
print ("Recall:", recall)


# In[89]:


clf = SVC(max_iter=1000, kernel = 'linear', cache_size=500)
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
acuracia = accuracy_score(labels_test, predictions)
precisao = precision_score(labels_test, predictions)
recall = recall_score(labels_test, predictions,average = 'macro')
print ("acuracia:", acuracia)
print ("Precisão:", precisao)
print ("Recall:", recall)


# CONCLUSÃO

# Após testar os classificadores com o algoritmo StratifiedShuffleSplit fornecido para teste, concluimos que o melhor aproveitamento seria com o uso do algoritmo Decision Tree Classifier, com as features e configurações abaixo:
# 
# Features:  'poi','salary', 'bonus', 'from_poi_fraction', 'to_poi_fraction'
# Configuração de parametros: criterion = 'entropy', min_samples_split= 6, max_depth=5
# 
# Apesar do uso dos algoritmos Select Kbest e GridSearchCV serem altamente indicados, obtive melhores resultados ao usar a intuição para escolha de features e estudo e muita tentativa e erro para paramentros.
# 
# Decision Tree Classifier:
# 
# Accuracy: 0.76636 - Precision: 0.38024 - Recall: 0.331  - F1: 0.35392   - F2: 0.33980 
# 
# Total predictions: 11000    	    
# True positives:  535   
# False positives:  960  
# False negatives: 1465  
# True negatives: 8040 		
# 
# 
# Os resultados acima mostram que:
# 
# Precisão: a probabilidade de que uma pessoa identificada como um POI seja realmente um POI é de 38%, e temos 62% de chances de uma pessoa classificada como POI ser inocente, um falso positivo.
# 
# Recall: a probabilidade do identificador sinalizar um POI no conjunto de testes é de 33%, e 67% das vezes o POI não seria rotulado.
# 
# Os resultados estão dentro do minimo esperado, mas não são excelentes.
# 
# Para melhorar a analise, acredito que deveria analisar mais os dados de email e/ou dados financeiros, talvez criando features com 'total payments', 'stock values' e 'salary', ou analisar os textos para encontrar padrões nas conversas.
# 
# 
# 

# In[116]:


dump_classifier_and_data(clf, my_dataset, features_list)


# In[117]:


#!/usr/bin/python3

""" a basic script for importing student's POI identifier,
    and checking the results that they get from it 
 
    requires that the algorithm, dataset, and features list
    be written to my_classifier.pkl, my_dataset.pkl, and
    my_feature_list.pkl, respectively

    that process should happen at the end of poi_id.py
"""

import pickle
import sys
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import StratifiedShuffleSplit
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

PERF_FORMAT_STRING = "\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\tRecall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print("Warning: Found a predicted label not == 0 or 1.")
                print("All predictions should take value 0 or 1.")
                print("Evaluating performance for processed predictions:")
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print(clf)
        print(PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
        print(RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
        print("")
    except:
        print("Got a divide by zero when trying out:", clf)

CLF_PICKLE_FILENAME = "my_classifier.pkl"
DATASET_PICKLE_FILENAME = "my_dataset.pkl"
FEATURE_LIST_FILENAME = "my_feature_list.pkl"

def dump_classifier_and_data(clf, dataset, feature_list):
    pickle.dump(clf, open(CLF_PICKLE_FILENAME, "wb") )
    pickle.dump(dataset, open(DATASET_PICKLE_FILENAME, "wb") )
    pickle.dump(feature_list, open(FEATURE_LIST_FILENAME, "wb") )

def load_classifier_and_data():
    clf = pickle.load(open(CLF_PICKLE_FILENAME, "rb") )
    dataset = pickle.load(open(DATASET_PICKLE_FILENAME, "rb") )
    feature_list = pickle.load(open(FEATURE_LIST_FILENAME, "rb"))
    return clf, dataset, feature_list

def main():
    ### load up student's classifier, dataset, and feature_list
    clf, dataset, feature_list = load_classifier_and_data()
    ### Run testing script
    test_classifier(clf, dataset, feature_list)

if __name__ == '__main__':
    main()

