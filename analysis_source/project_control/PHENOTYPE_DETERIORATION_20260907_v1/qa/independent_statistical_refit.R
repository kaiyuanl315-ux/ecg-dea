#!/usr/bin/env Rscript
# Independent R GLM check using natural-spline bases and explicit cluster sandwich.
# Does not import or call the Python analysis implementation.
args <- commandArgs(trailingOnly=TRUE)
base <- normalizePath(if (length(args)) args[1] else ".")
ref <- read.csv(file.path(base,"results","associations.csv"), check.names=FALSE)
cfg <- jsonlite::fromJSON(file.path(base,"qa","input_qa.json"))$covariate_transform
phen <- read.csv(gzfile(file.path(base,"qa","phenotype_analytic_local.csv.gz")), check.names=FALSE)
icu <- read.csv(gzfile(file.path(base,"qa","icu_analytic_local.csv.gz")), check.names=FALSE)
stopifnot(nrow(ref)==44L)
checked <- list()
for (i in seq_len(nrow(ref))) {
  r <- ref[i,]
  d <- if (r$domain=="phenotype") phen else icu
  if (r$subset=="first_encounter") d <- d[d$first_parent_encounter==1,]
  if (r$subset=="landmark60") {
    d <- d[d$landmark60_eligible==1,]
    d[[r$endpoint]] <- d[[paste0("landmark60_",r$endpoint)]]
  }
  d <- d[d$covariates_observed==1 & !is.na(d[[r$endpoint]]),]
  d$score_z <- d[[paste0(r$score,"_z")]]
  d$y <- d[[r$endpoint]]
  covariates <- c("age",if (r$adjustment=="M2") c("temperature_c","resprate","o2sat","sbp"))
  terms <- c("score_z","gender_clean")
  for (v in covariates) {
    info <- cfg[[v]]
    term <- if (info$nonlinear) sprintf(
      "splines::ns(%s_imp, knots=%.17g, Boundary.knots=c(%.17g,%.17g))",
      v,info$knots[2],info$knots[1],info$knots[3]) else paste0(v,"_imp")
    terms <- c(terms,term)
  }
  formula <- as.formula(paste("y ~",paste(terms,collapse=" + ")))
  fit <- glm(formula,data=d,family=binomial(),control=glm.control(epsilon=1e-10,maxit=100))
  stopifnot(fit$converged)
  x <- model.matrix(fit)
  prob <- fitted(fit)
  bread <- solve(crossprod(x,x*as.vector(prob*(1-prob))))
  score <- x*as.vector(d$y-prob)
  cluster_score <- rowsum(score,d$subject_id,reorder=FALSE)
  g <- nrow(cluster_score); n <- nrow(x); k <- ncol(x)
  covariance <- bread %*% crossprod(cluster_score) %*% bread * g/(g-1)*(n-1)/(n-k)
  beta <- unname(coef(fit)["score_z"])
  se <- sqrt(covariance["score_z","score_z"])
  z <- beta/se
  checked[[i]] <- data.frame(domain=r$domain,endpoint=r$endpoint,score=r$score,
    adjustment=r$adjustment,subset=r$subset,n=n,patients=g,events=sum(d$y),
    beta=beta,se=se,or=exp(beta),ci_low=exp(beta-qnorm(.975)*se),
    ci_high=exp(beta+qnorm(.975)*se),p=2*pnorm(-abs(z)),
    beta_abs_diff=abs(beta-r$beta),se_abs_diff=abs(se-r$se),
    count_match=(n==r$n && g==r$patients && sum(d$y)==r$events),converged=fit$converged)
}
out <- do.call(rbind,checked)
out$q <- NA_real_
for (domain in c("phenotype","icu")) {
  pick <- out$domain==domain & out$score=="ecg" & out$adjustment=="M2" & out$subset=="all"
  out$q[pick] <- p.adjust(out$p[pick],method="BH")
}
stopifnot(all(out$count_match),max(out$beta_abs_diff)<1e-7,max(out$se_abs_diff)<1e-7)
write.csv(out,file.path(base,"qa","independent_R_associations.csv"),row.names=FALSE)
summary <- list(status="PASS",n_models=nrow(out),all_counts_match=all(out$count_match),
  max_abs_beta_difference=max(out$beta_abs_diff),max_abs_se_difference=max(out$se_abs_diff),
  spline_implementation="R splines::ns with median interior knot and 10th/90th percentile boundary knots",
  variance_implementation="Explicit patient-score sandwich; G/(G-1)*(N-1)/(N-P) finite-sample correction",
  R_version=R.version.string,all_fits_converged=all(out$converged),
  preserved_parent_imputed_values=TRUE,no_mortality_model_training=TRUE)
jsonlite::write_json(summary,file.path(base,"qa","independent_R_check.json"),auto_unbox=TRUE,pretty=TRUE,digits=17)
print(summary)
