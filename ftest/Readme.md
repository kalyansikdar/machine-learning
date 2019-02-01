Part 1
1) Use the last 4 digits of your ID.

2) Use the first 3 letters of your last name.3) Use the first 3 letters of your first name.

(1)+(2)+(3) = 10 letters. If there are redundant, move to the next available letter.

(B) Use the data handler to pick all data instances of these 10 English letters
(total 10*39 = 390) data instances to form a new dataset. This is from the handwritten
 digits data.

(C) Do K-means on this dataset. Obtain the confusion matrix.

(D) Use optimal bipartite matching to permute the confusion matrix. Compute the clustering accuracy.

Submit (B) data files. (C) Confusion matrix  (D) confusion matrix after optimal matching. (E) Accuracy. (E) Your codes.

Part 2:
(A) Use the dataset you generates in Part 1 (total 10 classes, 390 data instances).

(B) Compute F-statistic on all 320 features. List the F-score in the original feature order.

(C) From the computed F-score, select the top 100 features. Put these selected features into a new dataset. This dataset should be 100x390.

(D) Do K-means on this dataset. Computer confusion matrix.

(E) Do the optimal bipartite graph match to re-order the confusion matrix. Compute the accuracy.

(D,E) are similar to Exam 2 Part 1.

Submit (B) the F-score. (C) the dataset with selected features. (D) confusion matrix. (E) Optimally -reordered confusion matrix and accuracy. Your code to compute the F-score.
