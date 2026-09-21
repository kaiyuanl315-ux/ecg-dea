#!/usr/bin/env Rscript
# Independent input joins, restriction rules, natural-spline GLMs and sandwich SEs.
args <- commandArgs(trailingOnly=TRUE)
pkg <- normalizePath(args[1]); parent <- normalizePath(file.path(pkg,'..','PHENOTYPE_DETERIORATION_20260907_v1'))
read_parent <- function(name) read.csv(gzfile(file.path(parent,'derived',paste0(name,'.csv.gz'))),check.names=FALSE)
base <- read_parent('test_base'); ph <- read_parent('phenotypes'); ic <- read_parent('icu'); su <- read_parent('support')
for (other in list(ph,ic,su)) {
  stopifnot(identical(base$row_id,other$row_id),identical(base$file_name,other$file_name),
            identical(base$subject_id,other$subject_id),!anyDuplicated(other$row_id))
}
cfg <- jsonlite::fromJSON(file.path(parent,'qa','input_qa.json'))$covariate_transform
ref <- read.csv(file.path(pkg,'results','phenotype_restriction_associations.csv'),check.names=FALSE)
stopifnot(nrow(ref)==24)
data <- cbind(base,ph[,setdiff(names(ph),names(base)),drop=FALSE])
checked <- list()
for (i in seq_len(nrow(ref))) {
  r <- ref[i,]; keep <- data$diagnosis_observed==1
  if (r$subset %in% c('exclude_any_cardiac','exclude_any_cardiac_M1',
                     'exclude_any_cardiac_first_parent','exclude_any_cardiac_single_run')) {
    keep <- keep & data$ami==0 & data$heart_failure==0 & data$af_flutter==0
  } else if (r$subset=='exclude_AMI') keep <- keep & data$ami==0
  else if (r$subset=='exclude_HF') keep <- keep & data$heart_failure==0
  else if (r$subset=='exclude_AF') keep <- keep & data$af_flutter==0
  else stopifnot(r$subset=='full_reference')
  if (r$subset=='exclude_any_cardiac_first_parent') keep <- keep & data$first_parent_encounter==1
  keep <- keep & data$covariates_observed==1 & !is.na(data[[r$endpoint]])
  keep[is.na(keep)] <- FALSE; d <- data[keep,]
  stopifnot(nrow(d)==r$n,length(unique(d$subject_id))==r$patients,sum(d[[r$endpoint]])==r$events)
  if (r$status!='ok') stop('A planned model was not estimable; independent review is required.')
  d$score_z <- d[[paste0(r$score,'_z')]]; d$y <- d[[r$endpoint]]
  covariates <- c('age',if (r$adjustment=='M2') c('temperature_c','resprate','o2sat','sbp'))
  terms <- c('score_z','gender_clean')
  for (v in covariates) {
    info <- cfg[[v]]
    term <- if (info$nonlinear) sprintf(
      'splines::ns(%s_imp, knots=%.17g, Boundary.knots=c(%.17g,%.17g))',
      v,info$knots[2],info$knots[1],info$knots[3]) else paste0(v,'_imp')
    terms <- c(terms,term)
  }
  fit <- glm(as.formula(paste('y ~',paste(terms,collapse=' + '))),data=d,
             family=binomial(),control=glm.control(epsilon=1e-10,maxit=100))
  stopifnot(fit$converged)
  x <- model.matrix(fit); prob <- fitted(fit)
  bread <- solve(crossprod(x,x*as.vector(prob*(1-prob))))
  patient_scores <- rowsum(x*as.vector(d$y-prob),d$subject_id,reorder=FALSE)
  g <- nrow(patient_scores); n <- nrow(x); k <- ncol(x)
  covariance <- bread %*% crossprod(patient_scores) %*% bread * g/(g-1)*(n-1)/(n-k)
  beta <- unname(coef(fit)['score_z']); se <- sqrt(covariance['score_z','score_z'])
  checked[[i]] <- data.frame(subset=r$subset,endpoint=r$endpoint,score=r$score,adjustment=r$adjustment,
    n=n,patients=g,events=sum(d$y),beta=beta,se=se,or=exp(beta),
    ci_low=exp(beta-qnorm(.975)*se),ci_high=exp(beta+qnorm(.975)*se),p=2*pnorm(-abs(beta/se)),
    beta_abs_difference=abs(beta-r$beta),se_abs_difference=abs(se-r$se))
}
out <- do.call(rbind,checked); out$q <- NA_real_
selected <- out$subset=='exclude_any_cardiac';out$q[selected] <- p.adjust(out$p[selected],method='BH')
stopifnot(max(out$beta_abs_difference)<1e-7,max(out$se_abs_difference)<1e-7,
          max(abs(out$q[selected]-ref$q[selected]))<1e-7)
write.csv(out,file.path(pkg,'qa','independent_R_associations.csv'),row.names=FALSE)

# Descriptive population membership and all stratum counts are independently reconstructed.
bref <- read.csv(file.path(pkg,'results','mortality_negative_care_escalation.csv'),check.names=FALSE)
counts <- list()
for (i in seq_len(nrow(bref))) {
  r <- bref[i,]; kind <- if (grepl('^icu',r$endpoint)) 'icu' else if (grepl('^imv',r$endpoint)) 'imv' else 'pressor'
  source <- if (kind=='icu') ic else su
  flag <- if (kind=='icu') 'eligible' else paste0(kind,'_eligible')
  keep <- base$death_72h==0 & source[[flag]]==1
  outcome <- source[[r$endpoint]]
  if (r$subset=='first_parent24') keep <- keep & base$first_parent_encounter==1
  if (r$subset=='landmark60_24') {
    flag <- if (kind=='icu') 'landmark60_eligible' else paste0(kind,'_landmark60_eligible')
    column <- if (kind=='icu') 'landmark60_icu24' else paste0(kind,'_landmark60_24')
    keep <- keep & source[[flag]]==1;outcome <- source[[column]]
  }
  keep[is.na(keep)] <- FALSE
  stratum <- ifelse(base$fusion<0.456303122639656,'Low',ifelse(base$fusion<0.6162797212600708,'Intermediate','High'))
  stopifnot(identical(stratum,base$risk_stratum))
  if (r$effect=='proportion') {
    if (r$stratum_or_contrast!='Total') keep <- keep & stratum==r$stratum_or_contrast
    n <- sum(keep); e <- sum(outcome[keep]); g <- length(unique(base$subject_id[keep])); estimate <- e/n
  } else {
    numerator <- if (r$stratum_or_contrast=='High_minus_Low') 'High' else 'Intermediate'
    estimate <- mean(outcome[keep & stratum==numerator])-mean(outcome[keep & stratum=='Low'])
    n <- sum(keep);e <- sum(outcome[keep]);g <- length(unique(base$subject_id[keep]))
  }
  stopifnot(n==r$n,e==r$events,g==r$patients,abs(estimate-r$estimate)<1e-14)
  counts[[i]] <- data.frame(subset=r$subset,endpoint=r$endpoint,stratum_or_contrast=r$stratum_or_contrast,
    n=n,patients=g,events=e,estimate=estimate,absolute_difference=abs(estimate-r$estimate))
}
write.csv(do.call(rbind,counts),file.path(pkg,'qa','independent_R_event_counts.csv'),row.names=FALSE)
summary <- list(status='PASS',models_checked=nrow(out),new_models=21,parent_reference_models=3,
  max_beta_difference=max(out$beta_abs_difference),max_se_difference=max(out$se_abs_difference),
  descriptive_rows_checked=nrow(bref),all_counts_and_estimates_match=TRUE,
  independent_cohorts_from_parent_inputs=TRUE,splines='R natural splines with locked knots',
  variance='Explicit patient sandwich with G/(G-1)*(N-1)/(N-P)',R_version=R.version.string)
jsonlite::write_json(summary,file.path(pkg,'qa','independent_R_check.json'),auto_unbox=TRUE,pretty=TRUE,digits=17)
print(summary)
