# CS4341_Project3
## Lisa Spalding & Esteban Aranda
## 09/15/19

To run this project open the terminal window and cd to the correct directory.
To run type 'python Project3.py [input].csv [output].csv' where 'input' has to be
a specified csv file inside the directory with the data to analyze and 'output'
will be the name of the file where the actual result of the matches that were used
as test will be printed first and then the predictions from the Decision Tree
in the order that they were computed in.

Project3.py calls FeatureExtraction.py to obtain all the specified features
(which one can read about in 'features.txt' and in the report),
which are printed to 'features.csv' (as a side not the last column in
'features.csv' does not contain information about a feature but rather the
winner from each match), and then contains a Decision Tree and
Random Forest with 10-fold cross validation to predict the outcome of the
Connect-4 games. Both, the Decision Tree and the Random Forest are called
with all individual inputs, all features, and all features except one at a time.
All the data from both are printed in the terminal and additionally in csv
files named 'DTresults.csv' and 'RFresults.csv' respectively.