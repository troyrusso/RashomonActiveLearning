### Summary: Simulates data
### Inputs:
# N: Number of observations
# K: Number of covariates.
# ClassProportion: Proportion of each class
# MeanMatrix: Average of each covariate
# CorrVal: Correlation between variables
### Output:
# dat: A data set

### Notes:
#### Should replace be TRUE

GenerateDataFunc = function(N, K, CovCorrVal, NBins = NA){
  
  ### Validations ###
  # if(length(MeanMatrix) != K){
  #   print(paste0("The length of Betas has to be ", K ,"."))
  # }
  if(K == 1){
    print("Need at least two covariates.")
  }
  
  ### Correlation Matrix ###
  # if(length(VarCov) == 1){
  #   SigmaMatrix = diag(K)
  #   SigmaMatrix[SigmaMatrix==0] = VarCov}else if(length(VarCov) != 1){
  #     SigmaMatrix = VarCov
  #   }
  SigmaMatrix = diag(K)
  SigmaMatrix[1,2] = CovCorrVal
  SigmaMatrix[2,1] = CovCorrVal
  
  ### True Betas ###
  TrueBetas = matrix(rnorm(n = K, mean = 0, sd = 1), ncol = K)

  ### Predictors ###
  MeanMatrix = rep(0,K)
  X = MASS::mvrnorm(n = N,
                    mu = MeanMatrix,
                    Sigma = SigmaMatrix)
  if(!is.na(NBins)){
    X = apply(X, MARGIN = 2, FUN = function(x) ntile(x, NBins))
    }

  
  ### Probabilities ###
  Logit = X %*% t(TrueBetas)
  epsilon = rnorm(n = N, mean = 0, sd = 1)
  Y = Logit + epsilon
  
  Logit = cbind(0, Logit)
  Probs = exp(Logit)/rowSums(exp(Logit))
  
  ### Preliminary Data Set ###
  dat = data.frame(ID = 1:length(Y), Y = Y, epsilon = epsilon, X = X)
  
  ### Add Useless Covariate ###
  dat = cbind(dat, sample(1:NBins, size = nrow(dat),replace = TRUE))
  colnames(dat) = c("ID", "Y", "epsilon", paste0("X", 1:(K+1)))
  rownames(dat) = dat$ID
  
  ### Return ###
  return(list(dat = dat,
              TrueBetas = TrueBetas))
}