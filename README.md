# cs6350
Nolan Angerbauer for CS5350/6350 in University of Utah

To use the Decision Tree in the DecisionTree folder you need to import the DecisionTree class from decision_tree.py
When creating the object you only need the max depth as an input
You then call fit to build the tree and you give it training data with the features and labels seperated into two arrays and you give the function you want to use to split the data
The only options for the functions at this point is Entropy, Majority Error, and Gini Index
You can then use the tree with the predict function by giving it any non-labeled data