
from datetime import datetime

import preprocess5

startIdx = 5001
endIdx = 11701


print("startIdx:", startIdx, "endIdx", endIdx)
print(" Time start: "+ str(datetime.now()))
headerName = ['Id', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9']
pathString = "../../dataset/test"
train_features5 = preprocess5.preprocessForAll5('../../dataset/sample_submission.csv', pathString, startIdx, endIdx)
preprocess5.writeToCSV('../../output/test_features5_part2.csv', headerName, train_features5)

 