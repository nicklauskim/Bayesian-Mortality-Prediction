
library(data.table) # for data processing 
library(ROCR)       # for AUC calculation 
# *******************************************************
# *******************************************************
# LOAD DATA / PROCESS DATA / SET UP 
# *******************************************************
# *******************************************************
path = "C:/Users/ryanj/Documents/UCLA/STATS_202C/Project"
path = "C:/Users/RyanODell/Documents/UCLA/ucla_stats/Stats_202C/Project"
#path="C:/Users/jacsw/OneDrive/Documents/Stat 202C/Project"
# path = "~/Documents/Spring 2022/STATS 202C/Final Project"

setwd(path)

dat = "/heart_failure_clinical_records_dataset (1).csv"


DT = fread(paste0(path, dat))
str(DT)


# poster for logistic regression
DT[, time := NULL]



# use normal random variables for prior 
cols = names(DT)[names(DT) != "DEATH_EVENT" ]
X = as.matrix(DT[ ,  ..cols])

# outcome variables 
y = DT$DEATH_EVENT


# add intercept 
X = cbind(rep(1, nrow(X)), X)
colnames(X) = c("Intercept" , cols )

# subset x to smaller number of predictors
set.seed(100)
train.idx = sample(nrow(X), 0.75*nrow(X))


X.train = X[train.idx,]
X.test  = X[-train.idx,]

y.train = y[train.idx]
y.test  = y[-train.idx]

DT.train = DT[train.idx,]

cols = colnames(X)[c(1,2,3,5,6,9,12)]
cols

#******************************************************
#******************************************************
#*              Functions for Sampler 
#******************************************************
#******************************************************


leap_frog_scheme = function(gradient, 
                            step_size , 
                            leap_frog_steps,
                            position, 
                            momentum,
                            ...){
  
  # Function Arguments
  # gradient        --- function for gradient of log posterior density
  # step_size       --- step size of leap frog scheme
  # leap_frog_steps --- number of leap frog steps in inner loop
  # position        --- current position
  # momentum        --- current draw of momentum
  # ...             --- additional function arguments passed to log_posterior
  #                     and gradient functions
  
  # Outputs
  # list with proposed momentum and
  # proposed position for calculating
  # acceptance criteria
  
  for (i in 1:leap_frog_steps) {
    
    # do one step of leap frog 
    momentum_new1 = momentum  + 0.5 * gradient(position, ...)  * step_size
    position_new  = position  + step_size * momentum_new1
    momentum_new2 = momentum_new1 + 0.5 * gradient(position_new, ...) * step_size
    
    
    # recycle parameters for next step 
    position = position_new
    momentum = momentum_new2
    
  }
  
  # return list of position 
  # and momentum 
  return(list(position = position,
              momentum = momentum))
  
}



HMCMC = function(log_posterior, gradient, step_size, leap_frog_steps, init_pars, num_iters , ...) {
  
  # Function Arguments
  # log_posterior   --- function for calculating log posterior density
  # gradient        --- function for gradient of log posterior density
  # step_size       --- step size of leap frog scheme
  # leap_frog_steps --- number of leap frog steps in inner loop
  # init_pars       --- vector of starting values
  # num_iters       --- number of outer iterations (# of samples to draw)
  # ...             --- additional function arguments passed to log_posterior
  #                     and gradient functions
  
  # Outputs
  # matrix of dimensions (num_iters x len(init_pars))
  # of draws from the posterior distribution 
  
  # Initialize parameters
  
  # dimension of parameter space
  len = length(init_pars)
  
  # matrix of output values of our samples
  out = matrix(0, nrow = num_iters, ncol = len )
  
  # prepare for looping 
  current_position = init_pars
  for (i in 1:num_iters ) {
    
    # draw momentum 
    current_momentum = rnorm(len, sd = 5)
    
    # run leap on current position and moment we drew 
    leap_frog_output = leap_frog_scheme(gradient, 
                                        step_size,
                                        leap_frog_steps,
                                        current_position,
                                        current_momentum, 
                                        ...)
    
    # get proposed momentum and position
    proposed_position = leap_frog_output$position
    proposed_momentum = leap_frog_output$momentum
    
    # calculate modified acceptance criteria
    
    # log posteriors of proposed position / momentum
    prop = log_posterior(proposed_position, ...) +
      sum(dnorm(proposed_momentum, sd =5, log = T)) 
    # log posteriors of current position / momentum 
    previous = log_posterior(current_position, ...) +
      sum(dnorm(current_momentum, sd =5, log = T))
    
    # subtract them 
    a = prop - previous 
    
    # calculate acceptance (in log scale)
    if (log(runif(1)) < a) {
      # accept and recycle parameters 
      current_position = proposed_position
    }
    # store the value 
    out[i, ] = current_position 
  }
  
  # return the matrix of values 
  return(out)
}



# numerically stable link function implementation
sigmoid = function(beta, x){
  1 - 1/(1+exp(x %*% beta))
}




single_prediction_interval = function(x, posterior, n = 20 ,  type ){
  
  # Function arguments
  # x         --- new data point to generate posterior predictive dist
  # posterior --- matrix of posterior draws from MCMC sampler
  # n         --- number of simulations from the likelihood per
  #               draw of the posterior distribution
  # type      --- "dist" for entire distribution or 
  #               "mean" for just posterior average 
  
  # Outputs
  # outputs a vector of posterior predictive draws when type 
  # is dist
  # outputs a number of posterior average of the predictive 
  # distribution when type is mean 
  
  # use 1-1/(1+exp()) numerically stable
  # implementation of the link function 
  # in logistic regression
  posterior_param = 1 - 1/(1 +exp(posterior %*% x ) )
  
  # change to vector, dont want a matrix
  posterior_param = as.vector(posterior_param)
  
  # grab its length
  len = length(posterior_param)
  
  # simulate bernoulli trials
  y = rbinom( len , size = n, prob = posterior_param )
  
  # normalize them 
  out = y/n
  
  # return what we need 
  if( type == "dist"){
    return(out)
    
  }
  else if(type == "mean"){
    return(mean(out))
  }
  
  
}

# ************************************************
# ************************************************
# ************************************************
#
#          SIMULATIONS
#
# ************************************************
# ************************************************
# ************************************************


# ************************************************
# ************************************************
#
#  NORMAL PRIOR CODE 
#
# ************************************************
# ************************************************

# log posterior for normal prior 
log_post = function(beta, X  , y  ) {
  # POSTERIOR FUNCTION FOR LOGISTIC REGRESSION 
  
  # likelihood function
  p = 1  - 1 / (1 + exp( X%*%beta ))
  p = as.vector(p)
  
  log_lik = sum(dbinom(y, size=1, prob=p, log=TRUE))
  
  
  log_dprior = sum(dnorm(beta,  0 , 5 , log = TRUE  ))
  
  return(log_lik  + log_dprior )
}

# gradient of log posterior for normal prior 
grad_log_post = function(beta, X, y ){
  
  # logistic 
  err = as.vector(y-sigmoid(beta, X))
  # pointwise multiply / sum  + log prior 
  
  colSums( err * X) + 2/25 * beta
}


# get MLE for starting param 
ncols = c("DEATH_EVENT",cols[-1])

mle = coef(glm(DEATH_EVENT ~ . , data = DT.train[,..ncols], 
               family = "binomial"))

mle

posterior_draws = HMCMC(log_post,
                        grad_log_post,
                        step_size = 0.00025, # 0.0000005
                        leap_frog_steps  = 250,
                        init_pars = mle,
                        num_iters = 50000,
                        X = X.train[,cols],
                        y = y.train)
str(posterior_draws)

# add column names
colnames(posterior_draws) = cols

# posterior means 
apply(posterior_draws,2,mean)

# trace plots 
par(mfrow = c(2,4))
plot(posterior_draws[,1],
     type = "l",
     xlab = "Iteration",
     ylab = "",
     main = "Intercept")

plot(posterior_draws[,2],
     type = "l",
     xlab = "Iteration",
     ylab = "",
     main = "Age")

plot(posterior_draws[,3],
     type = "l",
     xlab = "Iteration",
     ylab = "",
     main = "Anaemia")

plot(posterior_draws[,4],
     type = "l",
     xlab = "Iteration",
     ylab = "",
     main = "Diabetes")

plot(posterior_draws[,5],
     type = "l",
     xlab = "Iteration",
     ylab = "",
     main = "Ejection Fraction")

plot(posterior_draws[,6],
     type = "l",
     xlab = "Iteration",
     ylab = "",
     main = "Serum Creatinine")

plot(posterior_draws[,7],
     type = "l",
     xlab = "Iteration",
     ylab = "",
     main = "Smoking")

par(mfrow = c(1,1))


# marginal posterior histograms 
hist(posterior_draws[,1])
hist(posterior_draws[,2])
hist(posterior_draws[,3])
hist(posterior_draws[,4])
hist(posterior_draws[,5])
hist(posterior_draws[,6])
hist(posterior_draws[,7])


# save posterior draws
save(posterior_draws, file= "sub_model_hmcmc.RData")
# load back into R if needed 
# load("sub_model_hmcmc.RData")

# prediction code for every data in test set 
preds = apply(X.test[,cols], 1 , single_prediction_interval,
              post = posterior_draws,
              n = 1000,
              type = "mean")

str(preds)

pred = ROCR::prediction(preds , y.test)
perf = ROCR::performance(pred , "auc")
perf@y.values
#  0.8008

# ************************************************
# ************************************************
#
# UNIFORM PRIOR CODE 
#
# ************************************************
# ************************************************

# log posterior for uniform prior 
log_post = function(beta, X  , y  ) {
  # POSTERIOR FUNCTION FOR LOGISTIC REGRESSION 
  
  # likelihood function
  p = 1  - 1 / (1 + exp( X%*%beta ))
  p = as.vector(p)
  
  #print(p)
  log_lik = sum(dbinom(y, size=1, prob=p, log=TRUE))
  
  
  # diffuse prior for theta 
  return(log_lik )
}

# gradient log posterior for uniform prior
grad_log_post = function(beta, X, y ){
  n = length(X)
  # logistic 
  err = as.vector(y-sigmoid(beta, X))
  # pointwise multiply / sum  + log prior 
  # diffuse prior 
  colSums( err * X) 
}



posterior_draws_new = HMCMC(log_post,
                            grad_log_post,
                            step_size = 0.00025, # 0.0000005
                            leap_frog_steps  = 250,
                            init_pars = mle,
                            num_iters = 50000,
                            X = X.train[,cols],
                            y = y.train)
str(posterior_draws_new)

# add column names 
colnames(posterior_draws_new) = cols

# trace plots 
plot(posterior_draws_new[,1],type = "l")
plot(posterior_draws_new[,2],type = "l")
plot(posterior_draws_new[,3],type = "l")
plot(posterior_draws_new[,4],type = "l")
plot(posterior_draws_new[,5],type = "l")
plot(posterior_draws_new[,6],type = "l")
plot(posterior_draws_new[,7],type = "l")

# posterior means 
apply(posterior_draws_new , 2, mean )

# posterior distributions 
hist(posterior_draws_new[,1])
hist(posterior_draws_new[,2])
hist(posterior_draws_new[,3])
hist(posterior_draws_new[,4])
hist(posterior_draws_new[,5])
hist(posterior_draws_new[,6])
hist(posterior_draws_new[,7])

# save the code outputs 
save(posterior_draws_new, file =  "sub_model_hmcmc_uniformative.RData")
# load back in for further analysis
#load("sub_model_hmcmc_uniformative.RData")

# test set predictions
preds2 = apply(X.test[,cols], 1 , single_prediction_interval,
               post = posterior_draws_new,
               n = 1000,
               type = "mean")

# get test set AUC 
pred = ROCR::prediction(preds2 , y.test)
perf = ROCR::performance(pred , "auc")
perf@y.values
# 0.7952


# ************************************************
# ************************************************
#
# LAPLACE PRIOR CODE 
#
# ************************************************
# ************************************************

# log posterior for laplace prior
log_post = function(beta, X  , y  ) {
  # POSTERIOR FUNCTION FOR LOGISTIC REGRESSION 
  
  # likelihood function
  p = 1  - 1 / (1 + exp( X%*%beta ))
  p = as.vector(p)
  
  #print(p)
  log_lik = sum(dbinom(y, size=1, prob=p, log=TRUE))
  
  
  #print(log_dprior)
  
  log_dprior = 0
  for(i in beta){
    log_dprior = log_dprior + ddexp(i,0,1 , log = TRUE)
  }
  # diffuse prior for theta 
  return(log_lik + log_dprior )
}

# laplace density function 
ddexp = function(x, mu, sig, log ){
  # density function for double exponential distribution 
  if(log== FALSE){
    1/(sqrt(2)*sig) * exp(-sqrt(2)/ sig * sum(abs(x - mu)))
  }else{
    -log(sqrt(2))-log(sig) + (-sqrt(2)/ sig)* sum(abs(x - mu))
  }
  
}


# gradient log prior for laplace distribution
grad_log_ddexp = function(x, mu , sig){
  
  sqrt(2)/ sig * sum( sign(x - mu ))
}


# gradient log posterior for laplace prior
grad_log_post = function(beta, X, y ){
  n = length(X)
  # logistic 
  err = as.vector(y-sigmoid(beta, X))
  # pointwise multiply / sum  + log prior 
  
  
  grad_prior = rep(0,length(beta))
  for(i in 1:length(beta)){
    grad_prior[i] = grad_log_ddexp(beta[i],0,1)
  }
  
  colSums( err * X)  + grad_prior
}



posterior_draws_3 =HMCMC(log_post,
                         grad_log_post,
                         step_size = 0.00025, # 0.0000005
                         leap_frog_steps  = 250,
                         init_pars = mle,
                         num_iters = 50000,
                         X = X.train[,cols],
                         y = y.train)
str(posterior_draws_3)

# add column names back 
colnames(posterior_draws_3) = cols

# trace plots 
plot(posterior_draws_3[,1],type = "l")
plot(posterior_draws_3[,2],type = "l")
plot(posterior_draws_3[,3],type = "l")
plot(posterior_draws_3[,4],type = "l")
plot(posterior_draws_3[,5],type = "l")
plot(posterior_draws_3[,6],type = "l")
plot(posterior_draws_3[,7],type = "l")

# posterior means 
apply(posterior_draws_3 , 2, mean )

# posterior histograms 
hist(posterior_draws_3[,1])
hist(posterior_draws_3[,2])
hist(posterior_draws_3[,3])
hist(posterior_draws_3[,4])
hist(posterior_draws_3[,5])
hist(posterior_draws_3[,6])
hist(posterior_draws_3[,7])

# save the samples
save(posterior_draws_3, file =  "sub_model_hmcmc_double_exp.RData")
# load back in for further analysis
#load( "sub_model_hmcmc_double_exp.RData")

# predictions for out of sample data 
preds3 = apply(X.test[,cols], 1 , single_prediction_interval,
               post = posterior_draws_3,
               n = 1000,
               type = "mean")
# auc calculations 
pred = ROCR::prediction(preds3 , y.test)
perf = ROCR::performance(pred , "auc")
perf@y.values
# 0.8104

# **************************************
# **************************************
# **************************************
#
# PREDICTIVE DISTRIBUTION on test sets
#
# **************************************
# **************************************
# **************************************

# person who lived example
y.test[75]
X.test[75,]

# compare the distributions
# normal prior 
dists = single_prediction_interval(X.test[75,cols],
                                   post = posterior_draws,
                                   n = 1000, 
                                   type = "dist")
# uniform prior 
dists2 = single_prediction_interval(X.test[75,cols],
                                    post = posterior_draws_new,
                                    n = 1000, 
                                    type = "dist")
# laplace prior 
dists3 = single_prediction_interval(X.test[75,cols],
                                    post = posterior_draws_3,
                                    n = 1000, 
                                    type = "dist")
# put them into a data frame 
dist.DT = data.table(normal = dists,
                     unif   = dists2,
                     dexp   = dists3)

# get density estimates 
normal_dist = density(dists, from = 0, to  = 1, n = 1000 )
unif_dist   = density(dists2, from = 0, to  = 1 , n = 1000)
dexp_dist   = density(dists3, from = 0, to  = 1 , n = 1000)


# library for latex expressions
# in R plot titles  / labels 
library(latex2exp) 

# plot the 3 density estimates 
plot(normal_dist ,lwd = 1, lty = 2,
     main = TeX(r'($P(Y_{new} |X_{new}, X, y) = \int P(Y_{new}|X_{new}, \theta) P(\theta |y,X)d\theta$)'), 
     xlab = "", xlim = c(0,.4))
lines(unif_dist , lwd = 1, lty = 1,col = "grey")
lines(dexp_dist , lwd = 1, lty = 1,col = "steelblue")
legend("topright", legend= c("Normal","Unif", "Laplace"),
       lwd = rep(1,3) , 
       lty = c(2,1,1),
       col = c("black","grey","steelblue"),
       cex = 0.65)




