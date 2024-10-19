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

GenerateDataFunc = function(N, K, NClass, CovCorrVal, NBins = NA){
  
  ### Validations ###
  # if(length(ClassProportion) != NClass){
  #   print(paste0("The length of ClassProportion has to be ", NClass ,"."))
  # }
  # if(round(sum(ClassProportion), 1e-10) != 1){
  #   print("The sum of ClassProportion has to be 1")
  # }
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
  
  ### Class Proportion ###
  ClassProportion = rep(1/NClass, NClass)
  
  ### True Betas ###
  TrueBetas = matrix(rnorm(n = K * (NClass - 1), mean = 0, sd = 1),
                     ncol = K)

  ### Predictors ###
  MeanMatrix = rep(0,K)
  X = MASS::mvrnorm(n = NClass*N,
                    mu = MeanMatrix,
                    Sigma = SigmaMatrix)
  if(!is.na(NBins)){
    X = apply(X, MARGIN = 2, FUN = function(x) ntile(x, NBins))
    }

  
  ### Probabilities ###
  Logit = X %*% t(TrueBetas)
  epsilon = rnorm(n = NClass*N, mean = 0, sd = 1)
  YStar = Logit + epsilon
  
  Logit = cbind(0, Logit)
  Probs = exp(Logit)/rowSums(exp(Logit))

  ### Labels ###
  Y = apply(Probs, 1, function(p) sample(1:NClass, size = 1, prob = p)) - 1
  
  ### Preliminary Data Set ###
  PrelimDat = data.frame(Y = as.factor(Y), YStar = YStar, epsilon = epsilon, X = X)
  
  ### Class Proportion ###
  NClassSamples = round(N * ClassProportion)
  NClassSamples[1] = NClassSamples[1]+ (N - sum(NClassSamples))
  IndicesList = lapply(X = 0:(NClass-1),
                       FUN = function(c) sample(x = rownames(PrelimDat[PrelimDat$Y == c, ]),
                                                size = NClassSamples[c+1],
                                                replace = TRUE)) 
  dat = cbind(ID = 1:N,PrelimDat[unlist(IndicesList), ])
  
  ### Add Useless Covariate ###
  dat = cbind(dat, sample(1:NBins, size = nrow(dat),replace = TRUE))
  colnames(dat) = c("ID", "Y", "YStar", "epsilon", paste0("X", 1:(K+1)))
  rownames(dat) = dat$ID
  
  ### Return ###
  return(list(dat = dat,
              TrueBetas = TrueBetas,
              noise = epsilon))
}
