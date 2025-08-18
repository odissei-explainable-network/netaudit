// generated with brms 2.22.0
functions {
}
data {
  int<lower=1> N;  // total number of observations
  vector[N] Y;  // response variable
  int<lower=0> Nmi;  // number of missings
  array[Nmi] int<lower=1> Jmi;  // positions of missings
  int<lower=1> K;  // number of population-level effects
  matrix[N, K] X;  // population-level design matrix
  int<lower=1> Kc;  // number of population-level effects after centering
  int<lower=1> K_phi;  // number of population-level effects
  matrix[N, K_phi] X_phi;  // population-level design matrix
  int<lower=1> Kc_phi;  // number of population-level effects after centering
  int prior_only;  // should the likelihood be ignored?
}
transformed data {
  matrix[N, Kc] Xc;  // centered version of X without an intercept
  vector[Kc] means_X;  // column means of X before centering
  matrix[N, Kc_phi] Xc_phi;  // centered version of X_phi without an intercept
  vector[Kc_phi] means_X_phi;  // column means of X_phi before centering
  for (i in 2:K) {
    means_X[i - 1] = mean(X[, i]);
    Xc[, i - 1] = X[, i] - means_X[i - 1];
  }
  for (i in 2:K_phi) {
    means_X_phi[i - 1] = mean(X_phi[, i]);
    Xc_phi[, i - 1] = X_phi[, i] - means_X_phi[i - 1];
  }
}
parameters {
  vector<lower=0,upper=1>[Nmi] Ymi;  // estimated missings
  real par_b_1;
  real par_b_2;
  real par_b_3;
  real par_b_4;
  real par_b_5;
  real par_b_6;
  real par_b_8;
  real par_b_10;
  real par_b_12;
  real par_b_13;
  real par_b_15;
  real par_b_16;
  real par_b_18;
  real par_b_19;
  real par_b_20;
  real par_b_21;
  real par_b_22;
  real par_b_24;
  real par_b_25;
  real par_b_26;
  real par_b_27;
  real par_b_28;
  real par_b_30;
  real Intercept;  // temporary intercept for centered predictors
  vector<lower=0>[Kc_phi] b_phi;  // regression coefficients
  real<lower=0> Intercept_phi;  // temporary intercept for centered predictors
}
transformed parameters {
  vector[Kc] b;  // regression coefficients
  real lprior = 0;  // prior contributions to the log posterior
  b[7] = 0;
  b[9] = 0;
  b[11] = 0;
  b[14] = 0;
  b[17] = 0;
  b[23] = 0;
  b[29] = 0;
  b[31] = 0;
  b[32] = 0;
  b[33] = 0;
  b[1] = par_b_1;
  b[2] = par_b_2;
  b[3] = par_b_3;
  b[4] = par_b_4;
  b[5] = par_b_5;
  b[6] = par_b_6;
  b[8] = par_b_8;
  b[10] = par_b_10;
  b[12] = par_b_12;
  b[13] = par_b_13;
  b[15] = par_b_15;
  b[16] = par_b_16;
  b[18] = par_b_18;
  b[19] = par_b_19;
  b[20] = par_b_20;
  b[21] = par_b_21;
  b[22] = par_b_22;
  b[24] = par_b_24;
  b[25] = par_b_25;
  b[26] = par_b_26;
  b[27] = par_b_27;
  b[28] = par_b_28;
  b[30] = par_b_30;
  lprior += normal_lpdf(b[1] | 0, 2.5);
  lprior += normal_lpdf(b[2] | 0, 2.5);
  lprior += normal_lpdf(b[3] | 0, 2.5);
  lprior += normal_lpdf(b[4] | 0, 2.5);
  lprior += normal_lpdf(b[5] | 0, 2.5);
  lprior += normal_lpdf(b[6] | 0, 2.5);
  lprior += normal_lpdf(b[8] | 0, 2.5);
  lprior += normal_lpdf(b[10] | 0, 2.5);
  lprior += normal_lpdf(b[12] | 0, 2.5);
  lprior += normal_lpdf(b[13] | 0, 2.5);
  lprior += normal_lpdf(b[15] | 0, 2.5);
  lprior += normal_lpdf(b[16] | 0, 2.5);
  lprior += normal_lpdf(b[18] | 0, 2.5);
  lprior += normal_lpdf(b[19] | 0, 2.5);
  lprior += normal_lpdf(b[20] | 0, 2.5);
  lprior += normal_lpdf(b[21] | 0, 2.5);
  lprior += normal_lpdf(b[22] | 0, 2.5);
  lprior += normal_lpdf(b[24] | 0, 2.5);
  lprior += normal_lpdf(b[25] | 0, 2.5);
  lprior += normal_lpdf(b[26] | 0, 2.5);
  lprior += normal_lpdf(b[27] | 0, 2.5);
  lprior += normal_lpdf(b[28] | 0, 2.5);
  lprior += normal_lpdf(b[30] | 0, 2.5);
  lprior += normal_lpdf(Intercept | 0, 2.5);
  lprior += exponential_lpdf(b_phi | 1);
  lprior += exponential_lpdf(Intercept_phi | 1);
}
model {
  // likelihood including constants
  if (!prior_only) {
    // vector combining observed and missing responses
    vector[N] Yl = Y;
    // initialize linear predictor term
    vector[N] mu = rep_vector(0.0, N);
    // initialize linear predictor term
    vector[N] phi = rep_vector(0.0, N);
    Yl[Jmi] = Ymi;
    mu += Intercept + Xc * b;
    phi += Intercept_phi + Xc_phi * b_phi;
    mu = inv_logit(mu);
    phi = exp(phi);
    target += beta_lpdf(Yl | mu .* phi, (1 - mu) .* phi);
  }
  // priors including constants
  target += lprior;
}
generated quantities {
  // actual population-level intercept
  real b_Intercept = Intercept - dot_product(means_X, b);
  // actual population-level intercept
  real b_phi_Intercept = Intercept_phi - dot_product(means_X_phi, b_phi);
}
