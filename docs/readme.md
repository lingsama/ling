# Recommender System for Stock Trading and Electronic Trading Alerts Digital Assistant with friendly GUI


# Downloading Data

Before running the python code, you'll first need to download all data we'll be using. 
This data is located in the Combined_News_DJIA1.csv. 
# 
Requirements and Installation

In order to run the Python code, you'll need the following libraries.


1. [TensorFlow](http://www.tensorflow.org/)
2. [NumPy](http://www.numpy.org/)

3. [Matplotlib](http://www.matplotlib.org/)

4. Installing [Anaconda Python](https://www.anaconda.com/) and [Keras](https://keras.io/)

5. [Sklearn](http://scikit-learn.org/stable/)

The easiest way to install keras 
(Backend is TensorFlow) as well as NumPy, Matplotlib, Sklearn and so on is to start with the Anaconda Python distribution.


Follow the installation instructions for Anaconda Python. 
We recommend using Python 3.6.

Follow the platform-specific TensorFlow installation instructions. 
Be sure to follow the "Installing with Anaconda" process, and create a Conda environment named tensorflow.


Deep Learning

**Sentiment Analysis** is a classic problem in RNN with LSTM model. 
We try to implement it in keras framework and we also prepared a tensorflow version. 
The best accuracy is about 53.9%.

# 
Statistical Machine Learning

**Random Forest**, **K Nearest Neighbors**, **SVM**, 
**Voting** , **Logistic Regression** are also used for this prediction. 
More details can be found in the homepage of [Scikit-learn Classification]
(http://scikit-learn.org/stable/supervised_learning.html#supervised-learning)


Algorithms | Accuracy
--------   | ---
Random Forest | 50.53%
KNN    | 55.82%
SVM     | 48.14%
Voting | 50.26%
Logistic Regression | 48.94%


#	   Long-short term memory
![](https://raw.githubusercontent.com/lingsama/ling/master/pictures/lstm6.png)
![structure of LSTM](https://raw.githubusercontent.com/lingsama/ling/master/pictures/lstm7.png)
## The core idea behind LSTMs
The key to LSTMs is the cell state, the horizontal line running through the top of the diagram.
The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged.
The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates.
![](https://raw.githubusercontent.com/lingsama/ling/master/pictures/lstm8.png)
![](https://raw.githubusercontent.com/lingsama/ling/master/pictures/lstm9.png)
![](https://raw.githubusercontent.com/lingsama/ling/master/pictures/lstm10.png)
![](https://raw.githubusercontent.com/lingsama/ling/master/pictures/lstm11.png)
![](https://raw.githubusercontent.com/lingsama/ling/master/pictures/lstm12.png)
#          Code
##         Machine learning code
<code>
    # -*- coding: utf-8 -*-
    
    # Import dependencies
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Read data
    df = pd.read_csv('Combined_News_DJIA1.csv', parse_dates=True, index_col=0)
    #print('read csv')
    # Plot class distribution
    sns.countplot(x='Label', data=df)
    #????
    #plt.show()
    
    # Check the data
    df.head()
    #print(df)
    #print('--------------------')
    # Select only the top N news to use as features of the classifier. N ranges from 1 to 25.
    # In this case, N = 20. 
    columns = ['Top' + str(i+1) for i in range(20)]
    #print (columns)
    
    # Create a new column with the Top N news joined.
    df['joined'] = df[columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    
    # Create a new dataframe with only Label and joined columns
    df1 = df[['Label', 'joined']].copy()
    
    # Take a look
    df1.head()
    #print(df1)
    
    # According to the author of the dataset, the data should be split as it:
    # Train set: from 2008-08-08 to 2014-12-31 
    # Test set: from 2015-01-02 to 2016-07-01
    #print('---------------------')
    train = df1.ix['2008-08-08':'2014-12-31'].shape
    test = df1.ix['2015-01-02':'2016-07-01'].shape
    #print('this is train:',train)
    #test(378,2)
    #print('this is test:',test)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Create a tfidf object. Remove english stop words and use 10000 features.
    vect = TfidfVectorizer(max_features=10000, stop_words='english')
    #print(vect)
    
    # Transform the joined column into a tfidf sparse matrix
    X = vect.fit_transform(df1['joined'])
    #print(X)
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer
    
    # Use tfidf followed by svd is known as lsa or lsi.
    svd = TruncatedSVD(1000)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    
    # Apply lsa
    X = lsa.fit_transform(X)
    #print(X)
    
    # Split data into train and test
    X_train = X[:1611]
    #print (X_train.shape)
    X_test = X[1611:]
    #print (X_test.shape)
    
    # Get labels
    y_train = df1['Label'].ix['2008-08-08':'2014-12-31']
    y_test = df1['Label'].ix['2015-01-02':'2016-07-01']
    
    #print(y_test)
    #print(type(y_test))
    
    
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC
    from sklearn.linear_model.logistic import LogisticRegression
    
    # Create for different classifiers. Random Forest, KNN, Support Vector Machine and an Ensemble of these 3.
    rf = RandomForestClassifier()
    knn = KNeighborsClassifier(n_neighbors=3)
    svm = LinearSVC()
    vc = VotingClassifier(estimators=[('rf',rf), ('knn',knn), ('svm',svm)])
    lf = LogisticRegression()
    
    list_rf = []
    list_knn = []
    list_svm = []
    list_vc = []
    list_lf = []
    list_i = []
    
    
    
    for i in range(0,5):
    print('this is',i,'time')
    # Train them
    rf.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    vc.fit(X_train, y_train)
    lf.fit(X_train, y_train)
    
    # Check the accuracies
    ac_rf = rf.score(X_test, y_test)
    ac_knn = knn.score(X_test, y_test)
    ac_svm = svm.score(X_test, y_test)
    ac_vc = vc.score(X_test, y_test)
    ac_lf = lf.score(X_test, y_test)
    
    list_rf.append(ac_rf)
    list_knn.append(ac_knn)
    list_svm.append(ac_svm)
    list_vc.append(ac_vc)
    list_lf.append(ac_lf)
    list_i.append(i+1)
    
    
    print ("Accuracy rf is", ac_rf)
    print ("Accuracy knn is", ac_knn)
    print ("Accuracy svm is", ac_svm)
    print ("Accuracy vc is", ac_vc)
    print ("Accuracy lf is", ac_lf)
    print('--------------------------')
    print('END')
    
    print('random forest:',list_rf)
    print('KNN:',list_knn)
    print('SVM:',list_svm)
    print('VC:',list_vc)
    print('LF:',list_lf)
    
    df_all = pd.DataFrame({"rf":list_rf,"knn":list_knn,"svm":list_svm,"vc":list_vc,"lf":list_lf,"i":list_i})
    print(df_all)
    sns.jointplot(x='i',y='rf',data=df_all)
    plt.show()

</code>

##         Long-Short Term Memory
<code>
# -*- coding: utf-8 -*-
    """
        Sentiment Analysis with LSTMs
    """
    
    import pandas
    import collections
    import keras
    from keras.models import Sequential
    from keras.models import h5py
    from keras.layers import Dense
    from keras.layers import Activation
    from keras.layers import SimpleRNN
    from keras.optimizers import RMSprop
    import numpy as np
    from keras.utils import np_utils
    import matplotlib.pyplot as plt
    from sklearn.utils import shuffle
    from sklearn import preprocessing  
    from keras.utils import np_utils
    import tensorflow as tf
    from keras.layers import LSTM
    from keras.layers import Masking
    import numpy as np
    from sklearn.model_selection import train_test_split
    from keras.layers.embeddings import Embedding
    
    wordsList = np.load('wordsList.npy')
    print('Loaded the word list!')
    wordsList = wordsList.tolist() #Originally loaded as numpy array
    wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
    wordVectors = np.load('wordVectors.npy')
    print ('Loaded the word vectors!')
    print(len(wordsList))
    print(wordVectors.shape)
    
    #for testing
    baseballIndex = wordsList.index('baseball')
    print(wordVectors[baseballIndex])
    
    #for testing
    maxSeqLength = 10 #Maximum length of sentence
    numDimensions = 300 #Dimensions for each word vector
    firstSentence = np.zeros((maxSeqLength), dtype='int32')
    firstSentence[0] = wordsList.index("i")
    firstSentence[1] = wordsList.index("thought")
    firstSentence[2] = wordsList.index("the")
    firstSentence[3] = wordsList.index("movie")
    firstSentence[4] = wordsList.index("was")
    firstSentence[5] = wordsList.index("incredible")
    firstSentence[6] = wordsList.index("and")
    firstSentence[7] = wordsList.index("inspiring")
    #firstSentence[8] and firstSentence[9] are going to be 0
    print(firstSentence.shape)
    print(firstSentence) #Shows the row index for each word
    #with tf.Session() as sess:
    #    print(tf.nn.embedding_lookup(wordVectors,firstSentence).eval().shape)
        
    
    # Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
    import re
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    
    def cleanSentences(string):
        string = string.lower().replace("<br / >", " ")
        return re.sub(strip_special_chars, "", string.lower())
    
    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " ", string)
        string = re.sub(r"\)", " ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = re.sub(r",", " ", string)
        string = re.sub(r"'", " ", string)
        string = re.sub(r"\[", " ", string)
        string = re.sub(r"\]", " ", string)
        return string.strip().lower()
    
    # load dataset
    def get_data(file_name, drop_list):
        data = pandas.read_csv(file_name)
        ValueErrorCounter=0
        numWords = []
        word_freqs = collections.Counter()  #words frequency
        y_data1 = data[["Label"]]
        y_data = y_data1.values
        y_data = keras.utils.to_categorical(y_data)
        x_data_df = data.drop(drop_list,axis=1)
        x_data = x_data_df.values
        numFiles = len(x_data[:,0])
        numTops = len(x_data[0,:])
                             
        maxSeqLength = 600
        ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
        fileCounter = 0
        
        for i in range(numFiles):#numFiles
            line=''
            for j in range(numTops):#numTops
                try:
                    line=line+' '+x_data[i,j][1:]
                except TypeError:
                    print(x_data[i,j],i,j)
                    continue
            #print(line)
            numWords.append(len(line.split()))
            indexCounter = 0
            cleanedLine = cleanSentences(line)#cleanSentences
            split = cleanedLine.split()
            for word in split:
                #print(word)
                try:
                    ids[fileCounter][indexCounter] = wordsList.index(word)
                    word_freqs[word] += 1
                except ValueError:
                    word_freqs[word] = 1
                    ValueErrorCounter+=1
                    #print(word)
                    ids[fileCounter][indexCounter] = 400000-1 #Vector for unkown words
                indexCounter = indexCounter + 1
                if indexCounter >= maxSeqLength:
                    break
            fileCounter=fileCounter+1
        
        import matplotlib.pyplot as plt
        plt.hist(numWords, 50)
        plt.xlabel('Sequence Length')
        plt.ylabel('Frequency')
        plt.axis([100, 750, 0, 150])
        plt.show()
        print('The number of value error counter', ValueErrorCounter)
        print('The total number of files is', numFiles)
        print('The total number of words in the files is', sum(numWords))
        print('The average number of words in the files is', sum(numWords)/len(numWords))
        print('The different words is', len(word_freqs))
        
        np.save('X_Matrix', ids)
        np.save('Y_Matrix', y_data)
        return len(word_freqs), numWords, data.values, ids, y_data
    
    
    drop_list = ['Date','Label'] #drop attributes
    
    # get data
    #len_word_freqs, numWords, data, x_data, y_data = get_data("Combined_News_DJIA1.csv", drop_list) # you can comment this line for speed
    
    print('Getting data done!!')
    x_data = np.load('X_Matrix.npy')
    y_data = np.load('Y_Matrix.npy')
    
    #build rnn model lstm
    
    # IF YOU WANT TO USE THE SAME INITIAL WEIGHTS
    #np.random.seed(1337) 
    #tf.set_random_seed(1337)
    
    # 0.2 test data 0.8 train data
        
    #Xtrain, Xtest, ytrain, ytest = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    #Xtrain=Xtrain.reshape(-1,600,1)
    #Xtest=Xtest.reshape(-1,600,1)
    #len_word_freqs = 43747
    #
    #EMBEDDING_SIZE=128
    #nb_classes = 2
    #learning_rate = 1e-3
    #batch_size = 32
    #nb_epochs = 1000
    #hidden_units = 64
    #
    #early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, verbose=0, patience=120, mode='auto')
    #checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=str("./best_model"), monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    #
    #model = Sequential()
    #vocab_size= 44000
    #
    #model.add(Embedding(input_dim=10000, output_dim=300, input_length=maxSeqLength))
    ##model.add(Masking(mask_value=0, input_shape=Xtrain.shape[1:]))
    #
    ##        model.add(SimpleRNN(output_dim=hidden_units,
    ##                    activation='relu',
    ##                    input_shape=Xtrain.shape[1:], dropout=0))
    #model.add(LSTM(hidden_units, dropout=0.2, recurrent_dropout=0.2))
    #model.add(Dense(1))
    #model.add(Activation("sigmoid"))
    #model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
    #model.fit(Xtrain, ytrain, batch_size, nb_epochs,validation_data=(Xtest, ytest))
    #
    ##model.load_weights(str("./best_model")) 
    #
    #def recall(y_true, y_pred):
    #    index = np.where(y_true[:,1] == 1)
    #    #print "index",index
    #    positive_sample = y_true[index]
    #    predicts_sample = y_pred[index]
    #    true_positive = predicts_sample[np.where(predicts_sample[:,1] >= 0.5)]
    #    #print (len(true_positive))
    #    #print (len(positive_sample))
    #    return len(true_positive)/len(positive_sample)
    #
    #score, acc = model.evaluate(Xtest, ytest, batch_size)
    #y_pred = model.predict(Xtest)
    #best_recall = recall(ytest, y_pred)
    #
    #print("\nTest score: %.3f, accuracy: %.3f, recall: %.3f" % (score, acc, best_recall))
    #print('{}   {}      {}'.format('Predict','Truth','Sentence'))
    #for i in range(3):
    #    idx = np.random.randint(len(Xtest))
    #    xtest = Xtest[idx].reshape(1,40)
    #    ylabel = ytest[idx]
    #    ypred = model.predict(xtest)[0][0]
    #    sent = " ".join([wordsList[x] for x in xtest[0] if x != 0])
    #    print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))
    #Accuracy: 53.9
    
    
    maxSeqLength = 600
    numDimensions = 300 #Dimensions for each word vector 
    batchSize = 24
    lstmUnits = 64
    numClasses = 2
    iterations = 100000
    tf.reset_default_graph()
    labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
    
    data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors,input_data)
    
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
    
    
    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)
    
    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    
    import datetime
    sess = tf.InteractiveSession()
    
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)
    
    
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    from random import randint
    
    def get_Train_Test_Batch():
        Xtrain, Xtest, ytrain, ytest = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
        return Xtrain[0:batchSize], Xtest[0:batchSize], ytrain[0:batchSize], ytest[0:batchSize]
    
    
    for i in range(iterations):
        #Next Batch of reviews
        Xtrain, Xtest, ytrain, ytest = get_Train_Test_Batch();
        sess.run(optimizer, {input_data: Xtrain, labels: ytrain})
       
        #Write summary to Tensorboard
        if (i % 50 == 0):
            summary = sess.run(merged, {input_data: Xtrain, labels: ytrain})
            writer.add_summary(summary, i)
    
        #Save the network every 10,000 training iterations
        if (i % 10000 == 0 and i != 0):
            save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
            print("saved to %s" % save_path)
    writer.close()
     
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('models'))
    
    iterations = 10
    for i in range(iterations):
        Xtrain, Xtest, ytrain, ytest = get_Train_Test_Batch();
        print("Accuracy for this batch:", (sess.run(accuracy, {input_data: Xtest, labels: ytest})) * 100)
    #Accuracy for this batch: 47.5
</code> 