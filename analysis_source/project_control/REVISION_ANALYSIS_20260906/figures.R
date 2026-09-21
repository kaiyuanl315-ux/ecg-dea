#!/usr/bin/env Rscript
# All plotting and export use R; inputs are aggregate diagnostic tables.
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(jsonlite)})
args <- commandArgs(trailingOnly=TRUE)
pkg <- if(length(args)) normalizePath(args[1]) else normalizePath("project_control/REVISION_ANALYSIS_20260906")
inp <- file.path(pkg,"batch1_verified")
out <- file.path(pkg,"figures")
dir.create(out,showWarnings=FALSE,recursive=TRUE)
b <- read.csv(file.path(inp,"calibration_bins.csv"))
m <- read.csv(file.path(inp,"calibration_metrics.csv"))
t <- read.csv(file.path(inp,"threshold_metrics.csv"))
keep <- c("internal_test","external_article_saved")
b <- b[b$cohort %in% keep,]
t <- t[t$cohort %in% keep,]
labs <- c(internal_test="Internal test",external_article_saved="External: manuscript scores")
colours <- c("Raw score"="#777777","Validation-calibrated"="#287A9E",
             "Death detected"="#287A9E","False alert"="#B7BCC3")
theme_set(theme_classic(base_size=8,base_family="Helvetica") +
  theme(axis.line=element_line(linewidth=.35),axis.ticks=element_line(linewidth=.3),
        axis.text=element_text(size=7,colour="#303030"),plot.title=element_text(size=9,face="bold"),
        plot.subtitle=element_text(size=7),legend.position="bottom",legend.title=element_blank(),
        plot.tag=element_text(size=10,face="bold"),plot.caption=element_text(size=7,hjust=0),
        plot.margin=margin(8,8,8,8)))
percent <- function(x) paste0(format(round(x*100,1),trim=TRUE),"%")
val <- function(co,state,metric) m$estimate[m$cohort==co & m$score_state==state & m$metric==metric]
save_plot <- function(p,name,height=150) {
  w<-183/25.4;h<-height/25.4
  svglite::svglite(file.path(out,paste0(name,".svg")),width=w,height=h);print(p);dev.off()
  grDevices::pdf(file.path(out,paste0(name,".pdf")),width=w,height=h,family="Helvetica",useDingbats=FALSE);print(p);dev.off()
  ragg::agg_png(file.path(out,paste0(name,".png")),width=w,height=h,units="in",res=600);print(p);dev.off()
}
panel <- function(co,state) {
  d<-b[b$cohort==co & b$score_state==state,]
  xmax<-if(state=="raw") .8 else .10
  label<-if(state=="raw") "Raw model score" else "Validation-calibrated probability"
  ggplot(d,aes(mean_predicted,observed_rate))+
    geom_abline(slope=1,intercept=0,linetype=2,colour="#999999",linewidth=.4)+
    geom_line(colour=colours[if(state=="raw") "Raw score" else "Validation-calibrated"],linewidth=.5)+
    geom_point(size=1.7,shape=if(state=="raw") 16 else 17,
               colour=colours[if(state=="raw") "Raw score" else "Validation-calibrated"])+
    scale_x_continuous(labels=percent)+scale_y_continuous(labels=percent)+
    coord_fixed(xlim=c(0,xmax),ylim=c(0,xmax))+
    labs(title=labs[[co]],subtitle=sprintf("Brier %.5f | O/E %.2f",val(co,state,"brier"),val(co,state,"observed_expected_ratio")),
         x=label,y="Observed mortality proportion")
}
fig1 <- (panel(keep[1],"raw") | panel(keep[2],"raw")) /
        (panel(keep[1],"calibrated") | panel(keep[2],"calibrated")) +
  plot_annotation(tag_levels="a",title="Calibration of saved fusion scores",
    caption="Calibration fitted only on internal validation (17,889 encounters; 145 deaths).\nTest: 35,948 encounters / 302 deaths; external: 290,920 / 3,619.\nPoints: descriptive rates in fixed validation-score deciles; dashed line: ideal calibration.\nPanel scales differ before/after calibration. External score provenance remains a limitation.")
save_plot(fig1,"Figure_B1_calibration",174)

t$scenario <- paste(unname(labs[t$cohort]),ifelse(t$threshold_name=="low","Lower cut-off","Higher cut-off"),sep="\n")
t$scenario <- factor(t$scenario,levels=rev(unique(t$scenario)))
burden <- rbind(data.frame(scenario=t$scenario,component="Death detected",value=t$deaths_detected_per_1000),
                data.frame(scenario=t$scenario,component="False alert",value=t$false_alerts_per_1000))
burden$component<-factor(burden$component,levels=c("False alert","Death detected"))
pa <- ggplot(burden,aes(value,scenario,fill=component))+
  geom_col(width=.56)+scale_fill_manual(values=colours)+
  geom_text(data=t,aes(x=alerts_per_1000+7,y=scenario,label=sprintf("%.1f",alerts_per_1000)),
            inherit.aes=FALSE,hjust=0,size=2.6)+
  scale_x_continuous(expand=expansion(mult=c(0,.13)))+
  labs(title="Alert burden",x="Alerts per 1,000 encounters",y=NULL)
pb <- ggplot(t,aes(false_negative_rate*100,scenario))+
  geom_errorbar(aes(xmin=false_negative_rate_ci_lower*100,xmax=false_negative_rate_ci_upper*100),
                orientation="y",width=.14,linewidth=.45)+
  geom_point(shape=21,fill="#287A9E",size=2)+
  scale_x_continuous(limits=c(0,80),breaks=seq(0,80,20))+
  labs(title="Deaths below each cut-off",x="Missed deaths per 100 deaths",y=NULL)+
  theme(axis.text.y=element_blank(),axis.ticks.y=element_blank())
fig2 <- (pa|pb)+plot_layout(widths=c(1.35,1))+
  plot_annotation(tag_levels="a",title="Fixed-threshold errors and alert burden",
    caption="Lower/higher raw cut-offs: 0.45630312 / 0.61627972; calibrated equivalents: 0.00610022 / 0.03996975.\nCalibration does not change classifications. Error bars: patient-cluster bootstrap 95% CIs (2,000 replicates).\nThresholds are evaluated separately; alerts are hypothetical, not actual clinical interventions.\nExternal results use manuscript scores; alternative saved-score analysis is reported separately.")
save_plot(fig2,"Figure_B2_alert_burden",126)

cmp<-read.csv(file.path(pkg,"paired_comparison/model_metrics_with_ci.csv"))
cmp<-cmp[cmp$model!="fusion_original_score_sensitivity",]
model_names<-c(fusion="Fusion model",clinical_logistic="Clinical logistic",clinical_xgboost="Clinical XGBoost",clinical_ecg_xgboost="Clinical + ECG XGBoost")
cmp$model_label<-factor(unname(model_names[cmp$model]),levels=rev(unname(model_names)))
cmp$is_fusion<-cmp$model=="fusion"
comparison_panel<-function(co,met) {
  d<-cmp[cmp$cohort==co & cmp$metric==met,]
  p<-ggplot(d,aes(estimate,model_label))+
    geom_errorbar(aes(xmin=ci_lower,xmax=ci_upper),orientation="y",width=.15,linewidth=.4)+
    geom_point(aes(fill=is_fusion),shape=21,size=2)+scale_fill_manual(values=c("FALSE"="#B7BCC3","TRUE"="#287A9E"),guide="none")+
    labs(title=if(co=="internal") "Internal test" else "External: manuscript scores",
         x=if(met=="auroc") "AUROC (95% CI)" else "Average precision (95% CI)",y=NULL)
  if(met=="auroc") p<-p+scale_x_continuous(limits=c(.65,.95),breaks=c(.65,.75,.85,.95))
  if(met=="average_precision") p<-p+scale_x_continuous(limits=c(0,.15),breaks=seq(0,.15,.05))
  if(co=="external") p<-p+theme(axis.text.y=element_blank(),axis.ticks.y=element_blank())
  p
}
fig3<-(comparison_panel("internal","auroc") | comparison_panel("external","auroc")) /
      (comparison_panel("internal","average_precision") | comparison_panel("external","average_precision"))+
  plot_annotation(tag_levels="a",title="Stronger structured-data comparators",
    caption="XGBoost tuning: 3-fold patient-grouped CV within training only (4 configurations per model).\nClinical variables: 6; conventional ECG variables: PR, QRS, QT intervals and QRS axis.\nAll models predict actual death labels. CIs: 2,000 patient-cluster bootstrap replicates.\nExternal AUROC difference, fusion minus clinical + ECG: 0.0075 (95% CI -0.0018 to 0.0160).\nThese are secondary, unadjusted comparisons; external fusion-score provenance remains unresolved.")
save_plot(fig3,"Figure_B3_stronger_comparators",150)

write_json(list(backend="R",r_version=R.version.string,width_mm=183,dpi=600,
                figures=c("Figure_B1_calibration","Figure_B2_alert_burden","Figure_B3_stronger_comparators"),
                source="batch1_verified aggregate tables",bin_intervals="none; descriptive bins, not confidence bands",
                limitations="No claims of clinical benefit or safety; external score provenance caveat retained"),
           file.path(out,"FIGURE_SPEC.json"),pretty=TRUE,auto_unbox=TRUE)
cat("Exported 3 figures in SVG, PDF and 600 dpi PNG.\n")
