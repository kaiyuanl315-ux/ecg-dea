#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(data.table);library(jsonlite);library(sandwich)})
pkg<-normalizePath("project_control/REVISION_ANALYSIS_20260906")
root<-normalizePath(file.path(pkg,"../.."))
v<-fread(file.path(root,"data/proba_val_72.csv"),select=c("death_72h","proba_mean","subject_id"))
locked<-fromJSON(file.path(pkg,"batch1_verified/CALIBRATOR_AND_THRESHOLDS_FROZEN.json"))
clip<-function(x) pmin(1-1e-8,pmax(1e-8,x))
x<-qlogis(clip(v$proba_mean));y<-v$death_72h
fit<-glm(y~x,family=binomial(),control=glm.control(epsilon=1e-11,maxit=100))
dif<-max(abs(coef(fit)-c(locked$model$intercept,locked$model$slope)))
stopifnot(dif<1e-8)
cov<-vcovCL(fit,cluster=v$subject_id,type="HC1")
ci<-cbind(coef(fit)-qnorm(.975)*sqrt(diag(cov)),coef(fit)+qnorm(.975)*sqrt(diag(cov)))
ci_dif<-max(abs(ci-locked$model$coefficient_ci95_cluster_robust))
stopifnot(ci_dif<1e-6)
tm<-fread(file.path(pkg,"batch1_verified/threshold_metrics.csv"))
cm<-fread(file.path(pkg,"batch1_verified/calibration_metrics.csv"))
checks<-list()
for(co in c("internal_test","external_article_saved","external_original_score_sensitivity")) {
  if(co=="internal_test") {
    d<-fread(file.path(root,"data/proba_test_72.csv"),select=c("death_72h","proba_mean"));y<-d$death_72h;p<-d$proba_mean
  } else {
    sc<-if(co=="external_article_saved") "proba_mean_2" else "proba_mean_1"
    d<-fread(file.path(root,"data/zs/中山结果_v2.csv"),select=c("death_72h_or_triage",sc));y<-d$death_72h_or_triage;p<-d[[sc]]
  }
  pc<-plogis(locked$model$intercept+locked$model$slope*qlogis(clip(p)))
  py_bs<-cm[cohort==co & score_state=="calibrated" & metric=="brier",estimate]
  stopifnot(abs(mean((pc-y)^2)-py_bs)<1e-12)
  for(th in c("low","high")) {
    q<-p>=locked$thresholds_raw[[th]];row<-tm[cohort==co & threshold_name==th]
    counts<-c(tp=sum(q&y==1),fp=sum(q&y==0),tn=sum(!q&y==0),fn=sum(!q&y==1))
    stopifnot(all(counts==unlist(row[,.(tp,fp,tn,fn)])))
  }
  checks[[co]]<-list(brier_match=TRUE,all_confusion_counts_match=TRUE)
}
write_json(list(independent_implementation="R stats::glm + sandwich::vcovCL + direct tabulation",
                validation_coefficients_max_absolute_difference=dif,
                validation_cluster_ci_max_absolute_difference=ci_dif,
                cohorts=checks,status="PASS"),file.path(pkg,"INDEPENDENT_R_CHECK.json"),pretty=TRUE,auto_unbox=TRUE)
cat("Independent R recalculation: PASS\n")
