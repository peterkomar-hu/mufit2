data {
	int<lower=0> N;  // number of data points
	real x[N];  // data
}

parameters {
	
	simplex[2] w;

	// normal component
	real mu;
	real<lower=0> sigma;

	// Cauchy component
	real m;
	real<lower=0> s;
}

model {
	real lps[2];
	for (n in 1:N) {
		lps[1] = log(w[1]) + normal_lpdf(x[n] | mu, sigma);
		lps[2] = log(w[2]) + cauchy_lpdf(x[n] | m, s);
		
		target += log_sum_exp(lps);
	}
}
