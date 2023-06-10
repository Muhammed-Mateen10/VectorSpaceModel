#Ir assignmnet 2
import io
import os.path
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import nltk

import math

#uncomment these lines when compiling first time
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

lem = WordNetLemmatizer()  #LEMMATIZATION TOOL



class VSM:
  stopwords = []        #stopwords
  documents = 448        #total no of given abstracts
  VIndex = {}            #Index
  alpha = 0.001          #threshold value
  def __init__(self):
    self.getStopWords() #getting stopwords from document
    self.read()        #would read index from file if exists otherwise create index
  def getStopWords(self):
    stoplistpath = "/mnt/d/Academics/6th/IR/Assignment#2/provided/Stopword-List.txt"
    f = io.open(stoplistpath , "r" , encoding = 'cp1252')
    for line in f:
      for sp in line.split():
        self.stopwords.append(sp)
    f.close()
  def read(self):
    if os.path.isfile('/mnt/d/Academics/6th/IR/Assignment#2/index.txt'):  #if file exists(index exists)
      f = open('/mnt/d/Academics/6th/IR/Assignment#2/index.txt', 'r')
      self.VIndex = eval(f.read())      #eval converts string to dict
      f.close()
    else:
      self.createIndex()
  def createIndex(self):
    for i in range(1 , self.documents + 1):
      docpath = "/mnt/d/Academics/6th/IR/Assignment#2/static/Abstracts/" + str(i) + ".txt" #opening every document  one by one
      f = io.open(docpath , "r" , encoding = 'cp1252')
      for line in f:
        for word in line.split():
          word = word.strip(".,?:)(%'1234567890#$!{}") #removing there characters if they are at the begining
          word = word.replace("/" , " ")        #cluster/classification => claster classification
          word = word.replace("-" , " ")        
          tokenized_word = word_tokenize(word)    #chopping on spaces
          for word in tokenized_word:
            if(word not in self.stopwords):
              word = word.lower()
              obj = Lemmatizer(word)          #self defined object for adding part of speech tag to the word for better lemmatization
              word = obj.lemmatize()
              if(word not in self.VIndex):
               self.VIndex[word] = {'tf' : [0] * self.documents , 'df' : 0 , 'idf' : 0 , 'tf-idf' : [0] * self.documents} #tf : Term Frequency of a term in doc di , df : document frequency(No of documents containing this term . idf : log(n / df) , tf-idf : weighting scheme)
               self.VIndex[word]['tf'][i - 1] += 1 
               self.VIndex[word]['df'] = 1
               self.VIndex[word]['idf'] = math.log(self.documents / 2 , 10)
              else:
                if(self.VIndex[word]['tf'][i - 1] == 0): #first appearance of term in doc
                   self.VIndex[word]['df'] += 1
                self.VIndex[word]['tf'][i - 1] += 1
    self.calculateTfIdf()     
    self.write()
  def calculateTfIdf(self):
    for term in self.VIndex:       #calculation idf for every term
      self.VIndex[term]['idf'] = math.log(self.documents / (self.VIndex[term]['df']) , 10)
      for i in range(self.documents): #calculation tf-idf for every term
        self.VIndex[term]['tf-idf'][i] = self.VIndex[term]['tf'][i] * self.VIndex[term]['idf']
  def write(self):                #storing index on disk
    f = open('/mnt/d/Academics/6th/IR/Assignment#2/index.txt' , 'w')
    f.write(str(self.VIndex))          #writing dict->string format in file
    f.close()
  def getQueryVector(self , query):     #geting query vector <ti ... tn> 
    qv = {}
    query = word_tokenize(query)
    for word in query:
      if(not (word in self.stopwords)): #stop words doesnot contribute to the score
        word = word.lower()
        obj = Lemmatizer(word)
        word = obj.lemmatize()          #lemmatizing user's query term
        if(word in self.VIndex):
          if(word not in qv):
            qv[word] = {'tf' : 0 , 'tf-idf' : 0}
            qv[word]['tf'] = 1
          else:
            qv[word]['tf'] += 1
    self.calculateQueryVectorIdf(qv)      #scoring is based on tf-idf of a term
    dv = self.getDocumetnVector(qv.keys())
    return self.cosineScore( qv , dv)
  def calculateQueryVectorIdf(self , queryvector):
    for word in queryvector:
      queryvector[word]['tf-idf'] = queryvector[word]['tf'] * self.VIndex[word]['idf']
  def getDocumetnVector(self , query):   # calculating document Vector for calculating cosine similarity later
    doc = {}
    for i in range(0 , self.documents):
      doc[i] = {}
      for word in query:
        doc[i][word] = self.VIndex[word]['tf-idf'][i]
    return (doc)
  def cosineScore(self , qv , dv):     #qv = query vector represented as <t1 , t2 , ...tn> dv = document vector di = <tf-idf1st term , tf-idf .. nth term>
    filteredDocs = []
    for i in range(0 , self.documents):
      cosine = 0
      dp = self.dotproduct(qv , dv ,i) # dp repesents dot product of documnet di with query 
      magnitude = self.magnitude(qv , dv , i)
      if dp != 0 and magnitude != 0:
        cosine = (dp / magnitude)
      if(cosine > self.alpha):     #Filtering documents on alpha value gven 0.001
        filteredDocs.append((cosine, i + 1))
    return filteredDocs
  def dotproduct(self , qv , dv , docid):
    sum = 0
    for word in qv:
      if  qv[word]['tf-idf'] > 0 and  dv[docid][word] > 0: #accumulating score of dot product
        sum += qv[word]['tf-idf'] * dv[docid][word] #accumulating score of dot product
    return (sum)
  def magnitude(self , qv , dv , i):
    qm = 0
    dm = 0

    for word in qv:
      qm += math.pow( qv[word]['tf-idf'],2) 
      dm += math.pow( dv[i][word] , 2 )
    qm = math.sqrt(qm)    #magnitude formula
    dm = math.sqrt(dm)
    
    return qm * dm
  def runQuery(self , input):
    unrankedDocs = self.getQueryVector(input) #getting result of query (Initially unranked)
    ud = [doc for score , doc in unrankedDocs]
    unrankedDocs.sort(reverse = True) #sorting on decreasing order of cosine score
    print("Documents Retrieved:")
    print(ud)
    ranked = [doc for score , doc in unrankedDocs]
    print('Rank Of Document:')
    print(ranked)
    return [ud , ranked]
class Lemmatizer():    #self defined class for more precise lemmatization(adding pos (part of speech) tags to every word)
  def __init__(self , word):
    self.word = word
  def pos_tagger(self , nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None

  def lemmatize(self):
    pos_tagged =  nltk.pos_tag(nltk.word_tokenize(self.word)) 
    wordnet_tagged = list(map(lambda x: (x[0], self.pos_tagger(x[1])), pos_tagged))
    for word, tag in wordnet_tagged:
      if tag is None:
        return (word)
      else:       
        return (lem.lemmatize(word, tag))  #returning lemmatized word