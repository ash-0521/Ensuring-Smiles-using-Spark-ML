# Databricks notebook source
# MAGIC %md
# MAGIC ###Import all the Libraries

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.functions import col, when, max

from pyspark.ml import Pipeline

from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, RandomForestClassifier, LinearSVC, GBTClassifier, FMClassifier

from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create a Spark-Submit Session

# COMMAND ----------

PYSPARK_CLI = True # conditional statement to run only at shell
if PYSPARK_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

# Limit the log
spark.sparkContext.setLogLevel("WARN")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Load the sample dataset from DBFS

# COMMAND ----------

# Oracle BDCE
#csv = spark.read.csv('/user/agupta25/project/benefits1.csv', inferSchema=True, header=True)
# File location and type
file_location = "/FileStore/tables/benefits1.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)
  
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare the Data

# COMMAND ----------

df = df.select('BusinessYear', 'StateCode', 'IssuerId', 'SourceName', 'IsEHB', 'QuantLimitOnSvc', 'Exclusions', 'EHBVarReason',col("IsCovered").alias("label"))

df.show()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ###count the null values from prediction col

# COMMAND ----------

from pyspark.sql.functions import col, sum

# assuming that `df` is a Spark DataFrame and `label` is a column in `df`
null_count = df.select(sum(col("label").isNull().cast("integer"))).collect()[0][0]
print(null_count)


# COMMAND ----------

# MAGIC %md
# MAGIC ###Replace null or whitespace values with None. Later drop the values.

# COMMAND ----------

from pyspark.sql.functions import when, col

# Replace empty strings or whitespace with null values
df = df.withColumn('label', when(col('label').isin('', ' '), None).otherwise(col('label')))

# Drop null values from label column
df = df.dropna(subset=['label'])
df.show()



# COMMAND ----------

# MAGIC %md
# MAGIC ###Take Max of all the other columns in dataset having null values

# COMMAND ----------

df.agg({'IsEHB': 'max','QuantLimitOnSvc':'max','Exclusions':'max','EHBVarReason':'max'}).collect()


# COMMAND ----------

# MAGIC %md
# MAGIC ###Populating the aggregated values of other columns inplace of null values

# COMMAND ----------

df = df.fillna({
    "BusinessYear": 0,
    "StateCode": "",
    "IssuerId": 0,
    "SourceName": "",
    "IsEHB": "Yes",
    "QuantLimitOnSvc": "Yes",
    "Exclusions": "in vitro fertilization and artificial insemination",
    "EHBVarReason": "Using Alternate Benchmark",
    "label": ""
})

df.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ###Convert the label into 0 and 1 for classification modelling and prediction.

# COMMAND ----------

df = df.withColumn("label", when(df["label"] == "Covered", 1).otherwise(0))
df.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ###Shows the summary of dataset

# COMMAND ----------

df.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Shows the existing null values in dataset

# COMMAND ----------

df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split the Data for training & testing

# COMMAND ----------

splits = df.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
train_rows = train.count()
test_rows = test.count()
print("Training Rows:", train_rows, " Testing Rows:", test_rows)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the Pipeline
# MAGIC A predictive model often requires multiple stages of feature preparation. For example, it is common when using some algorithms to distingish between continuous features (which have a calculable numeric value) and categorical features (which are numeric representations of discrete categories). It is also common to *normalize* continuous numeric features to use a common scale (for example, by scaling all numbers to a proportinal decimal value between 0 and 1).
# MAGIC
# MAGIC A pipeline consists of a a series of *transformer* and *estimator* stages that typically prepare a DataFrame for
# MAGIC modeling and then train a predictive model. In this case, you will create a pipeline with seven stages:
# MAGIC - A **StringIndexer** estimator that converts string values to indexes for categorical features
# MAGIC - A **VectorAssembler** that combines categorical features into a single vector
# MAGIC - A **VectorIndexer** that creates indexes for a vector of categorical features
# MAGIC - A **VectorAssembler** that creates a vector of continuous numeric features
# MAGIC - A **MinMaxScaler** that normalizes continuous numeric features
# MAGIC - A **VectorAssembler** that creates a vector of categorical and continuous features
# MAGIC - A **DecisionTreeClassifier** that trains a classification model.

# COMMAND ----------

strIdx_SC = StringIndexer(inputCol = "StateCode", outputCol = "SC",handleInvalid='keep')
strIdx_SN = StringIndexer(inputCol = "SourceName", outputCol = "SN",handleInvalid='keep')
strIdx_EHB = StringIndexer(inputCol = "IsEHB", outputCol = "EHB",handleInvalid='keep')
strIdx_QL = StringIndexer(inputCol = "QuantLimitOnSvc", outputCol = "QL",handleInvalid='keep')
strIdx_EX = StringIndexer(inputCol = "Exclusions", outputCol = "EX",handleInvalid='keep')
strIdx_EHBVR = StringIndexer(inputCol = "EHBVarReason", outputCol = "EHBVR",handleInvalid='keep')


# the following columns are categorical number such as ID so that it should be Category features
catVect = VectorAssembler(inputCols = ["SC", "BusinessYear", "IssuerId", "SN", "EHB","QL","EHBVR"], outputCol="catFeatures")
catIdx = VectorIndexer(inputCol = catVect.getOutputCol(), outputCol = "idxCatFeatures", handleInvalid="skip")


# COMMAND ----------

# MAGIC %md
# MAGIC ###Shows the feature extraction count of cat features

# COMMAND ----------

# Fit the string indexers on the input data
strIdx_SC_model = strIdx_SC.fit(df)
strIdx_SN_model = strIdx_SN.fit(df)
strIdx_EHB_model = strIdx_EHB.fit(df)
strIdx_QL_model = strIdx_QL.fit(df)
strIdx_EX_model = strIdx_EX.fit(df)
strIdx_EHBVR_model = strIdx_EHBVR.fit(df)

# Transform the input data using the fitted string indexers
data_transformed = df
data_transformed = strIdx_SC_model.transform(data_transformed)
data_transformed = strIdx_SN_model.transform(data_transformed)
data_transformed = strIdx_EHB_model.transform(data_transformed)
data_transformed = strIdx_QL_model.transform(data_transformed)
data_transformed = strIdx_EX_model.transform(data_transformed)
data_transformed = strIdx_EHBVR_model.transform(data_transformed)

# Count the number of distinct values in each output column
distinct_counts = {
    "StateCode": data_transformed.select(countDistinct("SC")).collect()[0][0],
    "SourceName": data_transformed.select(countDistinct("SN")).collect()[0][0],
    "IsEHB": data_transformed.select(countDistinct("EHB")).collect()[0][0],
    "QuantLimitOnSvc": data_transformed.select(countDistinct("QL")).collect()[0][0],
    "Exclusions": data_transformed.select(countDistinct("EX")).collect()[0][0],
    "EHBVarReason": data_transformed.select(countDistinct("EHBVR")).collect()[0][0]
}

print(distinct_counts)

# COMMAND ----------

# cat feature vector is normalized

minMax = MinMaxScaler(inputCol = catIdx.getOutputCol(), outputCol="normFeatures")

featVect = VectorAssembler(inputCols=["normFeatures"], outputCol="features")

classification_models=["Logistic Regression (LR)","Decision Tree (DT)","Random Forest (RT)","Factorization Machine (FM)","Gradiest Boost (GBT)","Support Vector Machine (SVM)"]

#creating diff clasf algos for testing accuracy,computing time, precision, recall, ROC, PR
cls_mod=[]

cls_mod.insert(0,LogisticRegression(labelCol="label",featuresCol="features",maxIter=10,regParam=0.3,threshold=0.35)) 
cls_mod.insert(1,DecisionTreeClassifier(labelCol="label", featuresCol="features",seed=42)) 
cls_mod.insert(2,RandomForestClassifier(labelCol='label', featuresCol='features',seed=42)) 
cls_mod.insert(3,FMClassifier(labelCol='label', featuresCol='features', seed=42)) 
cls_mod.insert(4,GBTClassifier(labelCol='label', featuresCol='features', seed=42)) 
cls_mod.insert(5,LinearSVC(labelCol='label', featuresCol='features')) 

# COMMAND ----------

# define list of models made from Train Validation Split or Cross Validation
model = []
pipeline = []

# COMMAND ----------

# Pipeline process the series of transformation above, which is another transformation
for i in range(0,6):
    pipeline.insert(i,Pipeline(stages=[strIdx_SC,strIdx_SN,strIdx_EHB,strIdx_QL,strIdx_EX,strIdx_EHBVR, catVect, catIdx,minMax, featVect, cls_mod[i]]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tune hyperparameters using ParamGrid

# COMMAND ----------

paramGrid=[]

paramGrid.insert(0,(ParamGridBuilder() \
             .addGrid(cls_mod[0].regParam, [0.01, 0.3]) \
             .addGrid(cls_mod[0].elasticNetParam, [0.0, 0.5]) \
             .addGrid(cls_mod[0].maxIter, [10,20]) \
             .build()))
             
             
paramGrid.insert(1,ParamGridBuilder() \
             .addGrid(cls_mod[1].maxBins, [64,128,256]) \
             .addGrid(cls_mod[1].maxDepth, [2, 5, 10]) \
             .addGrid(cls_mod[1].impurity, ["gini", "entropy"]) \
             .addGrid(cls_mod[1].minInstancesPerNode, [1, 5, 10]) \
             .build())
             

paramGrid.insert(2,ParamGridBuilder() \
              .addGrid(cls_mod[2].numTrees, [50, 100, 150]) \
              .addGrid(cls_mod[2].maxBins, [64,128,256])
              .addGrid(cls_mod[2].maxDepth, [2, 5, 10]) \
              .build())


paramGrid.insert(3,ParamGridBuilder()\
.addGrid(cls_mod[3].regParam, [0.01, 0.1]) \
.addGrid(cls_mod[3].stepSize, [0.1,1])\
.addGrid(cls_mod[3].factorSize, [2,4])\
.build())


paramGrid.insert(4,ParamGridBuilder()\
.addGrid(cls_mod[4].maxDepth, [2, 5])\
.addGrid(cls_mod[4].maxIter, [10, 20])\
.addGrid(cls_mod[4].minInfoGain, [0.0])\
.build())

    
paramGrid.insert(5,ParamGridBuilder() \
             .addGrid(cls_mod[5].regParam, [0.01, 0.5]) \
             .addGrid(cls_mod[5].maxIter, [1, 5]) \
             .addGrid(cls_mod[5].tol, [1e-4, 1e-3]) \
             .addGrid(cls_mod[5].fitIntercept, [True, False]) \
             .addGrid(cls_mod[5].standardization, [True, False]) \
             .build())






# COMMAND ----------

# MAGIC %md
# MAGIC ### Used CrossValidator for modelling

# COMMAND ----------

cv=[]
K=3 
for i in range(0,6):
    cv.insert(i, CrossValidator(estimator=pipeline[i], 
                            evaluator=BinaryClassificationEvaluator(), 
                            estimatorParamMaps=paramGrid[i], 
                            numFolds=K))


#cv1 = CrossValidator(estimator=pipeline1, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid1, numFolds=K)
#cv2= CrossValidator(estimator=pipeline2, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid2, numFolds=K)
#cv3 = CrossValidator(estimator=pipeline3, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid3, numFolds=K)
#cv4 = CrossValidator(estimator=pipeline4, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid4, numFolds=K)
#cv5 = CrossValidator(estimator=pipeline5, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid5, numFolds=K)

#cv = TrainValidationSplit(estimator=pipeline, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)


# COMMAND ----------

# MAGIC %md
# MAGIC ###Calculating the computing time required to build a model

# COMMAND ----------

import time

start_time = []
end_time = []
computation_time = []

for i in range(0, 6):
    start_time.insert(i, time.time())
    model.insert(i, cv[i].fit(train))
    # model1 = cv1.fit(train)
    # model2 = cv2.fit(train)
    # model3 = cv3.fit(train)
    # model4 = cv4.fit(train)
    # model5 = cv5.fit(train)
    end_time.insert(i, time.time())
    computation_time.insert(i, (end_time[i] - start_time[i]) / 60.0)
    print("Computation time:",i," ",computation_time[i], "minutes")


# COMMAND ----------

import time

start_time = time.time()
model.insert(0, cv[4].fit(train))
    # model1 = cv1.fit(train)
    # model2 = cv2.fit(train)
    # model3 = cv3.fit(train)
    # model4 = cv4.fit(train)
    # model5 = cv5.fit(train)
end_time = time.time()
computation_time = ((end_time - start_time) / 60.0)
print("Computation time:",computation_time, "minutes")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the Pipeline Model
# MAGIC The model produced by the pipeline is a transformer that will apply all of the stages in the pipeline to a specified DataFrame and apply the trained model to generate predictions. In this case, you will transform the **test** DataFrame using the pipeline to generate label predictions.

# COMMAND ----------

prediction =[]
predicted =[]
for i in range(0,6):
    prediction.insert(i,model[i].transform(test))
    prediction[i].show()
    predicted.insert(i,prediction[i].select("features", "prediction","trueLabel"))
    predicted[i].show()
    
    

#LR
#prediction = model.transform(test)
#prediction.show(5)
#predicted = prediction.select("features", "prediction", "probability", "trueLabel")

#predicted.show(10, truncate=False)

#DT
#prediction1 = model1.transform(test)
#predicted1 = prediction1.select("features", "prediction", "probability", "trueLabel")

#predicted1.show(10, truncate=False)

#RF
#prediction2 = model2.transform(test)
#predicted2 = prediction2.select("features", "prediction", "probability", "trueLabel")

#predicted2.show(10, truncate=False)

#SVM
#prediction.insert(5,model[5].transform(test))
#predicted.insert(5, prediction[5].select("features", "prediction", "trueLabel"))

#prediction[5].show(10,truncate=False)
#predicted[5].show(10, truncate=False)

#GBT
#prediction4 = model4.transform(test)
#prediction4.show(5)
#predicted4 = prediction4.select("features", "prediction", "probability", "trueLabel")

#predicted4.show(10, truncate=False)

#FM
#prediction5 = model5.transform(test)
#predicted5 = prediction5.select("features", "prediction", "probability", "trueLabel")

#predicted5.show(10, truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC The resulting DataFrame is produced by applying all of the transformations in the pipline to the test data. The **prediction** column contains the predicted value for the label, and the **trueLabel** column contains the actual known value from the testing data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute Confusion Matrix Metrics
# MAGIC Classifiers are typically evaluated by creating a *confusion matrix*, which indicates the number of:
# MAGIC - True Positives
# MAGIC - True Negatives
# MAGIC - False Positives
# MAGIC - False Negatives
# MAGIC
# MAGIC From these core measures, other evaluation metrics such as *precision* and *recall* can be calculated.

# COMMAND ----------

precision=[]
recall=[]
metrics=[]

# COMMAND ----------

for i in range(0,6):
    tp = float(predicted[i].filter("prediction== 1.0 AND truelabel == 1").count())
    fp = float(predicted[i].filter("prediction== 1.0 AND truelabel == 0").count())
    tn = float(predicted[i].filter("prediction== 0.0 AND truelabel == 0").count())
    fn = float(predicted[i].filter("prediction==0.0 AND truelabel == 1").count())
    precision.insert(i,tp / (tp + fp))
    recall.insert(i,tp / (tp + fn))
    metrics.insert(i, spark.createDataFrame([
    ("TP", tp),
    ("FP", fp),
    ("TN", tn),
    ("FN", fn),
    ("Precision", tp / (tp + fp)),
    ("Recall", tp / (tp + fn))],["metric", "value"]))
    metrics[i].show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### View the Raw Prediction and Probability
# MAGIC The prediction is based on a raw prediction score that describes a labelled point in a logistic function. This raw prediction is then converted to a predicted label of 0 or 1 based on a probability vector that indicates the confidence for each possible label value (in this case, 0 and 1). The value with the highest confidence is selected as the prediction.

# COMMAND ----------

for i in range(0,6):
    prediction[i].select("rawPrediction", "prediction", "trueLabel").show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Calculating metrics such as ROC, PR, Accuracy, F1_score, Precision, Recall

# COMMAND ----------

evaluator = [None] * 6
ROC = [None] * 6
PR = [None] * 6
ev1 = [None] * 6
accuracy = [None] * 6
f1_score = [None] * 6

for i in range(0, 6):
    evaluator[i] = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="rawPrediction")
    ROC[i] = evaluator[i].evaluate(prediction[i], {evaluator[i].metricName: "areaUnderROC"})
    # print("ROC = {0:.3f}".format(auc_roc))

    PR[i] = evaluator[i].evaluate(prediction[i], {evaluator[i].metricName: "areaUnderPR"})
    # print("PR = {0:.3f}".format(auc_pr))

    ev1[i] = MulticlassClassificationEvaluator(labelCol='trueLabel', predictionCol='prediction')
    # accuracy
    accuracy[i] = ev1[i].evaluate(prediction[i], {evaluator[i].metricName: "accuracy"})
    # print("Accuracy = {0:.3f}".format(accuracy))

    # f1 score
    f1_score[i] = ev1[i].evaluate(prediction[i], {evaluator[i].metricName: "f1"})
    # print("F1 = {0:.3f}".format(f1_score))


# COMMAND ----------

# MAGIC %md
# MAGIC ###Comparing all metrics at one place

# COMMAND ----------

import pandas as pd

results = {
    'Model': classification_models,
    'Computation Time (min)': computation_time,
    'ROC': ROC,
    'PR': PR,
    'Accuracy': accuracy,
    'F1 Score': f1_score,
    'Precision': precision,
    'Recall': recall
}

df_results = pd.DataFrame.from_dict(results)
df_results = df_results.set_index('Model').transpose()

print(df_results)


# COMMAND ----------

# MAGIC %md
# MAGIC ###Calculate the Feature importance of the best Model - GBT Classifier.
# MAGIC

# COMMAND ----------

# Access feature importance of the best GBT model
bestModel = model[4].bestModel
gbtModel = bestModel.stages[-1]  # Assuming GBTClassifier is the last stage in the pipeline

featureImportances = gbtModel.featureImportances
# Create a list of tuples containing feature names and importances
importance_tuples = [(feature, importance) for feature, importance in zip(df.columns, featureImportances)]

# Sort the feature importances based on importance values in ascending order
sorted_importances = sorted(importance_tuples, key=lambda x: x[1], reverse=True)

# Print the feature importances in ascending order
print("Feature Importance:")
for feature, importance in sorted_importances:
    print("{}: {}".format(feature, importance))
