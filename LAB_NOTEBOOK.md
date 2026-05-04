5/2/2026, 9:04AM: Developed working numpy version of the project. Used an epoch count of 20 and a batch_size of 128. Tested Adam and SGD performance across different learning rates. Found that for each seed, Adam generalized better. Will change epoch count to 200.

5/2/2026, 1:01PM: Increased epoch count on numpy version to 200. Mean Adam test accuracy was only slightly better than SGD, achieving a generalization gap (difference between Adam and SGD; SGD Mean Accuracy - Adam Mean Accuracy = Generalization Gap) of 0.0018. Will change epoch count to 100 in attempt to reduce computation time.

5/2/2026, 2:25PM: Reduced numpy epoch count to 100. Computation time slightly reduced, but final results slightly worse. Generalization gap is -0.029, with Adam performing slightly better than SGD. Will change epoch count to 250.

5/2/2026, 4:53PM: Increased numpy epoch count to 250. Computation time significantly increased. Achieved generalization gap of 0.0018. Thus, results are indistinguishable from epoch count of 200. Will revert epoch count to 200. Will also change batch size to 1024 to reduce noise on plots.

5/2/2026, 6:58PM: Changed numpy batch size to 1024, achieved a generalization gap of 0.0001, with SGD performing slightly better on average. SGD test loss is less than Adam on seeds 0 and 3. Loss curves do not completely plateau, but noise was moderately decreased. Will increase epoch count to 300 and batch size to 2048.

5/2/2026, 9:11PM: Changed numpy epoch count to 300 and batch size to 2048. SGD test loss was less than Adam for every seed. Noticed smoother loss curves when lr (learning rate) = 0.001. Curves still do not completely plateau and noise was unaffected. Will increase epochs to 1000 and revert batch size to 1024.

5/3/2026, 8:57AM: Changed numpy epoch count to 1000 and batch size to 1024. As expected, noise on plot curves was mostly unaffected and computation time was significantly increased. Achieved greatest generalization gap of 0.0024, with SGD having the higher mean test accuracy. Across all seeds, 0.0003 and 0.05 were the best learning rates for Adam and SGD, respectively. Loss curves plateau at around 700 epochs. Will reduce epoch count to 700.

5/3/2026, 2:15PM: Reduced numpy epoch count to 600. No difference in generalization gap, mean accuracy, nor best learning_rate was observed. Computation time slightly reduced. SGD mean test and training error were both less than Adam mean test and training error respectively. Will now test PyTorch version.

5/3/2026, 8:40PM: Ran PyTorch version with epoch count of 600 and batch size of 1024. Achieved generalization gap of 0.0022 with SGD having a slightly higher mean test accuracy. Test error for SGD was less than Adam for all seeds. Across all seeds, 0.0001 and 0.05 were the best learning rates for Adam and SGD, respectively. Training accuracy was suspiciously high for some lr configurations (=1). Will decrease PyTorch epoch count to 500.

5/3/2026, 9:06PM: Reduced PyTorch epoch count to 500. Still received training accuracies of 1. Achieved generalization gap of 0.0004. Across all seeds, 0.001 and 0.05 were the best learning rates for Adam and SGD, respectively. Set epochs in ablation loop to 200. Computation time significantly increased. For the optimizer ablation, a β1 value of 0.05 produced the best results. For the MLP ablation, a deep-wide network produced the best result.
