
from datetime import datetime

import preprocess5

startIdx = 25000
endIdx = 31071


print("startIdx:", startIdx, "endIdx", endIdx)
print(" Time start: "+ str(datetime.now()))
headerName = ['Id', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9']
pathString = "../../dataset/train"
train_features5 = preprocess5.preprocessForAll5('../../dataset/train.csv', pathString, startIdx, endIdx)
preprocess5.writeToCSV('../../output/train_features5_part6.csv', headerName, train_features5)

 