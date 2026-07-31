do teams' historical performance data take into account the calibre of the team they were playing against? e.g. cabo verde lost to argentina but argentina are a great team

- k-NN is not very useful at predicting the outcomes of future matches because it only takes into account ELO diff which the CTMCs already take into account. So the signals are linearly dependent. This is called feature collinearity. However it would be very useful for predicting live, in-play matches. i could test the effectiveness of knn at predicting futuer matches by doing feature wrapper techniques and doing leave one out cross validation, getting the model to predict match outcomes.

- when calculating a statistic to measure variance between my model's predictions and the true bookie odds, I can't do direct comparison on decimal odds. Since decimal odds are non-linear, the difference between 1.10 and 1.20 (difference: 9.1%) is mathematically treated the same as 5.10 and 5.20 (difference: 2.0%), even thought the percentage differences are wildly different.

- current rendition assumes vig is distributed uniformly when in reality it is distributed more towards the underdogs and draws since favourites already have odds so low that people won't be tempted to bet on them

- we will turn the M>=10,000 historical match database into a vectorised form offline.

- i first used degree 2 Bernstein polynomials for dictating the weights of the different data, but i found
  exponential decay is a better fit since we want live match data to make a faster, more powerful impact.

- i can just copy over the scraper scripts from the gpu computer onto the epl scraper project

- still going to use the same tri-modal bayesian inference model to predict results of in-play matches

for 3-way markets, some sources say Shin's method is better and some say power law method is better, so i'll run A/B tests to see which does a better job.
