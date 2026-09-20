#!/usr/bin/env Rscript
# Independent support GLMs with natural spline basis and explicit cluster sandwich.
args <- commandArgs(trailingOnly=TRUE)
base <- normalizePath(args[1])
ref <- read.csv(file.path(base,"results","support_associations.csv"),check.names=FALSE)
cfg <- jsonlite::fromJSON(file.path(base,"qa","input_qa.json"))$covariate_transform
all_data <- read.csv(gzfile(file.path(base,"qa","support_analytic_local.csv.gz")),check.names=FALSE)
stopifnot(nrow(ref)==30L)
checked <- list()
for (i in seq_len(nrow(ref))) {
  r <- ref[i,]
  kind <- if (startsWith(r$endpoint,"imv")) "imv" else "pressor"
  hours <- sub(kind,"",r$endpoint,fixed=TRUE)
  d <- all_data[all_data[[paste0(kind,"_eligible")]]==1,]
  if (r$subset=="first_encounter") d <- d[d$first_parent_encounter==1,]
  if (r$subset=="landmark60") {
    d <- d[d[[paste0(kind,"_landmark60_eligible")]]==1,]
    d[[r$endpoint]] <- d[[paste0(kind,"_landmark60_",hours)]]
  }
  replacement <- switch(r$subset,raw_state=paste0("imv_raw_state",hours),
    no_dopamine=paste0("pressor_no_dopamine",hours),no_bolus=paste0("pressor_no_bolus",hours),NULL)
  if (!is.null(replacement)) d[[r$endpoint]] <- d[[replacement]]
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
  model <- glm(as.formula(paste("y ~",paste(terms,collapse=" + "))),data=d,
    family=binomial(),control=glm.control(epsilon=1e-10,maxit=100))
  stopifnot(model$converged)
  x <- model.matrix(model); prob <- fitted(model)
  bread <- solve(crossprod(x,x*as.vector(prob*(1-prob))))
  u <- x*as.vector(d$y-prob)
  cluster_score <- rowsum(u,d$subject_id,reorder=FALSE)
  g <- nrow(cluster_score); n <- nrow(x); k <- ncol(x)
  covariance <- bread %*% crossprod(cluster_score) %*% bread * g/(g-1)*(n-1)/(n-k)
  beta <- unname(coef(model)["score_z"]); se <- sqrt(covariance["score_z","score_z"])
  checked[[i]] <- data.frame(endpoint=r$endpoint,score=r$score,adjustment=r$adjustment,
    subset=r$subset,n=n,patients=g,events=sum(d$y),beta=beta,se=se,or=exp(beta),
    ci_low=exp(beta-qnorm(.975)*se),ci_high=exp(beta+qnorm(.975)*se),
    p=2*pnorm(-abs(beta/se)),beta_abs_diff=abs(beta-r$beta),se_abs_diff=abs(se-r$se),
    count_match=(n==r$n && g==r$patients && sum(d$y)==r$events),converged=model$converged)
}
out <- do.call(rbind,checked)
out$q <- NA_real_
primary <- out$score=="ecg" & out$adjustment=="M2" & out$subset=="all"
stopifnot(sum(primary)==4L)
out$q[primary] <- p.adjust(out$p[primary],method="BH")
stopifnot(all(out$count_match),max(out$beta_abs_diff)<1e-7,max(out$se_abs_diff)<1e-7)
write.csv(out,file.path(base,"qa","independent_R_support_associations.csv"),row.names=FALSE)
summary <- list(status="PASS",n_models=nrow(out),all_counts_match=all(out$count_match),
  max_abs_beta_difference=max(out$beta_abs_diff),max_abs_se_difference=max(out$se_abs_diff),
  separate_BH_family_size=sum(primary),R_version=R.version.string,
  spline_implementation="R splines::ns with original training percentile knots",
  variance_implementation="Explicit patient-score sandwich with G/(G-1)*(N-1)/(N-P) correction",
  all_fits_converged=all(out$converged),original_processed_covariates_preserved=TRUE,
  mortality_model_retraining=FALSE)
jsonlite::write_json(summary,file.path(base,"qa","independent_R_support_check.json"),auto_unbox=TRUE,pretty=TRUE,digits=17)
print(summary)
