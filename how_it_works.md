My own written explanation of how the entire project works (could later adapt this to a README)

We have scraped data from the Premier League 25/26 season every season all the way back to 11/12.
This data is information about every single ball event that happened in the match, for all of the ~5700 matches.

We want to use this data, along with some other predictive factors, to predict the probabilities of outcomes of matches that are currently in play. We choose in-play matches because they allow the use of predictive methods that cannot be used to predict future matches.

First of all, to reduce the data's dimensionality, we define a state space. For each event, the ball is in one of 12 states, capturing the ball's location in one of five discrete spatial pitch zones (Z:0 to Z:4) and the team with possession (P:H or P:A), plus two terminal absorbing goal states (Goal_H and Goal_A). Hence each Q matrix is 12x12.

15 seasons of matches are scraped for historical data, but to account for different tactical approaches throughout the years, we apply a exponential decay weight w\_{season} = e^{-\gamma\*\delta s} for older season data, where \gamma is a hyperparameter.

To predict in-play matches, we will run Monte Carlo simulations on a transition matrix Q_active, which is formed from a weighted combination of three other transition matrices Q_pre, Q_KNN, and Q_live. The weights of each Q at a time t are governed by continuous exponential decay curves with a half-life of 45 minutes, to ensure smooth weight decay without harsh, unnatural step functions.

- **Q_pre**: The prior Q matrix. To create this:
- We first calculate two separate global Q matrices, Q_home and Q_away, which are formed from the aggregated totals of the state transition counts and holding times of all the matches in the 15-season database.
- Then, if a team is playing at home, we take the global historical home n_ij and T_i values, scale them by a factor of alpha, and perform Bayesian conjugate updating to them using the club's specific n_ij and T_i values when also playing at home. The formula is

lambda\_{ij, updated} = \frac{alpha \* n_ij^global + n_ij^club}{alpha \* T_i^global + T_i^club}

- Vice versa for if the team's playing away. Alpha is the parameter that decides how highly we should weight general Premier League transition rates compared to specific club rates.
- After that, we take into account club strength rating differentials (from a standardised source like Opta). For each calculated rate (lambda_old), we scale it by a factor of e^(beta \* (+-)diff), with diff being positive or negative depending on whether the state transition is good (e.g. progressive pass) or bad (e.g. dispossession).
- This Q_pre matrix will be linearly combined with the next two matrices to create Q_active, which we will use to make the final predictions. The weight of Q_pre in creating Q_active will be dynamic; it is weighted highly at the start before the match is far underway, but drops off fast as we get a better idea of how the teams are playing on the day.

- **Q_KNN**: Nearest neighbours matches. For an in-play match currently at minute t, we want to be able to look back at past matches, find ones that were similar (at least) up until minute t, and then say, "Well, most matches that were similar to this one ended in this way, so it makes sense to predict the same for this live match too." To do this, we create 90 databases: one for each until-minute-t bucket for t in {1,2,... 90} (that is, every match in database until-minute-t is truncated to end at minute t, so its stats are only calculated until then). Then in each until-minute-t bucket database, we truncate all 5700 matches up until minute t, then calculate 5 stats from each truncated match: Field Tilt, Home Progression Ratio, Away Progression Ratio, Match Tempo, and Goal Diff, and normalise them. We then put them into a 5D vector and place them inside the database for the corresponding minute bucket. Then, if we want to find the most similar matches to an in-play one currently at minute t, we calculate the until-minute-t 5D vector for the current match, and look in database t and get the k closest neighbours based on Euclidean distance inside the 5D vector space. Thus Q_KNN is obtained by aggregating the observed values of n_ij and T_i from minute t to minute 90 from those k matches and calculating lambdas. Q_KNN's weight towards Q_active will be scaled based on how similar the neighbours are overall. (E.g., if the k nearest neighbours are such that they are all actually pretty far away but they're still the 'nearest', then this factor will receive a penalty. Maybe I'll scale it by the inverse or inverse square of the combined distances.) Generally, Q_KNN's weight is low at the start since we don't have a strong idea about how the match is being played, but is higher in the middle, when we have better information, and then drops off later as the CTMC reality dictates the contest.

- **Q_live**: Continuous-time Markov chain simulations. For the current match, we aggregate all the ball events, find how many times the ball entered a certain state and how long it was held in that state, then calculate the Poisson rates of transition to other states by dividing the counts by the holding times. This gives us an overview of how the ball moves around the pitch in this specific match, and this Q matrix is called Q_live. It is weighted low at the start as transition matrices are sparse, but high towards the end.

We connect the pipeline to a live feed of Opta event data from the match. The pipeline must execute start to finish in <200 ms to provide relevant, real-time probability updates. The things the pipeline must do in that time are:

- ingest the payload of event data
- update the k-nearest neighbours
- calculate the weights of the three matrices that combine to create Q_active
- runs short-horizon (initialising with the current ball state and scoreboard at time t, simulating from minute t to 90) Monte Carlo simulations on Q_active to calculate the updated probabilities

Additional notes:

- Dead time is capped at 15 seconds to prevent long lulls in play (VAR checks, injuries) from damaging the integrity of the match data.
- If a team hasn't visited a certain state yet, then its holding time will be zero, causing division by zero errors when blending matrices. The engine performs row-wise dynamic reallocation to prevent this by automatically falling back to priors or K-NN values for that state, and enforcing that every row of each Q matrix sums to zero.
- All simulation loops bypass CPU bottlenecks (like if/else statements and PCIe bus delays) by executing as pure PyTorch tensorised GPU bursts. This keeps latency well within our 200ms budget.

- As the final step, the engine simultaneously ingests live, in-play bookmaker odds from Bet365/Pinnacle, devigs them to find true market implied probabilities, and calculates real-time Market RMSE. Then the system flags positive EV arbitrage opportunities when the model judges the firms' predictions to be behind. The Q_KNN model will be particularly useful for discovering alpha.
