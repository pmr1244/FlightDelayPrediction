from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors


def parseLine(line):
	parts=line.split(',')
	label=int(parts[9])
	features=Vectors.dense([int(x) for x in parts[:9]])
	return LabeledPoint(label,features)

data=sc.textFile("/home/hduser/Desktop/spark/final_input_weather.csv")
parsedData=data.map(parseLine)


training=parsedData
test_data=sc.textFile("/home/hduser/Desktop/spark/test_data.csv")
test=test_data.map(parseLine)
model=NaiveBayes.train(training,1.0)
labelsAndPreds=test.map(lambda p:(model.predict(p.features),p.label))
accuracy=1.0*labelsAndPreds.filter(lambda(x,v):x==v).count()/test.count()


print("Accuracy= %f" % (accuracy))

