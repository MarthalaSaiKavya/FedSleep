# Implementation Details:

Dataset:

According to Fitbit, the overall sleep score is calculated by combining individual scores in sleep duration, sleep quality, and restoration, resulting in a total score of up to 100. Most individuals typically receive a score between 72 and 83. The sleep score ranges are as follows:
•	Excellent: 90-100
•	Good: 80-89
•	Fair: 60-79
•	Poor: Less than 60
Components of Sleep Score:
1.	Duration: Reflects the time spent asleep and awake. The more sleep, the higher the score.
2.	Quality: Considers the time spent in deep and REM sleep. More time in these sleep stages contributes to a higher score.
3.	Restoration: Considers factors such as sleeping heart rate and restlessness. A more relaxed sleep, with a lower heart rate and minimal tossing and turning, leads to a higher score.
In this analysis, the goal is to understand how to achieve a sleep score of 80 or more.
Dataset Overview:
The dataset comprises 8,522 samples from 4 users. The data has been split into 70:30 ratio for training samples for clients and validation samples on the server side. Additionally, an 80:20 split has been applied to the training samples of clients for training and testing purposes on each client.

Results:
The use of Federated Averaging reduced accuracy by 4.06% and an increase in the loss by 0.32. These results are substantial in the belief that federated learning can be used to converge models in a distributed-privacy-conserving fashion. The detailed client results have been presented in Table 1 and their learning progression Figure 1.

Figure 1: The above image shows the progression of the accuracy of individual clients after every aggregation. The image clearly shows the learnable capacity of the model.

Table 1: The table shows the performance of different clients over the course of learning. The table shows client-wise accuracy and loss values after each aggregation step.
