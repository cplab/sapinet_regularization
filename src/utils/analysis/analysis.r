# -------------- IMPORTS -------------- #
suppressPackageStartupMessages({
  library(docstring)
  library(data.table)
  library(ggplot2)
  library(lme4)
  library(emmeans)
  library(multcomp)
  library(doBy)
  library(dplyr)
})

# --------- INITIALIZATION --------- #
rm(list=ls(all=TRUE))  # clean up session variables.

source(file.path("snn", "utils", "analysis", "descriptive.r"))  # load descriptive statistics utility functions.
source(file.path("snn", "utils", "analysis", "plotting.r"))  # load plotting utility functions.

# --------- GLOBAL VARIABLES --------- #
# scan available run output directories.
AVAILABLE_RUNS <- sort(list.dirs(file.path("results"), recursive=FALSE, full.names=FALSE))

run_dirs <- "SciRep-FF-Final"
run_names <- run_dirs

meta_dir <- "SciRep-FF-Final"

# select smoothing level to represent the adaptive condition.
# .5 will be relabeled to "Scaled", -1 is "Uniform" (no adaptation).
adaptive_level <- .25
smoothing_levels <- c(adaptive_level, .5, -1)

# --------- UTILITY FUNCTIONS --------- #
read_data <- function(run, pattern="*.csv") {
  #' @title Reads tabluar Sapicore simulation output
  #'
  #' @description Processes tabular data files residing in the `run` directory being processed.
  #' Run can be specified by directory name or by index, with 0 representing the most recent timestamped run.
  #'
  #' @param run integer or string: a single run directory to process.
  #' @param pattern string: filename pattern to include in the search (e.g., "*.csv").
  #'
  #' @return data data.frame: a dataframe containing the data read from the directory.
  #' If multiple files exist in the directory, they will be concatenated.
  # current run data path; if provided with integers, starts from most recent.
  if (is.numeric(run)) {
    run_dir <- AVAILABLE_RUNS[length(AVAILABLE_RUNS)-run]
  } else {
    run_dir <- run
  }
  run_path <- file.path("results", run_dir)

  if (!dir.exists(run_path)) {
    stop(sprintf("Run directory does not exist (%s)", run_path))
  }
  message(sprintf("Processing run directory: %s", run_path))

  # read files and concatenate into a single master frame.
  files <- file.path(run_path, list.files(path=run_path, pattern=pattern, recursive=TRUE))
  frame <- data.frame(rbindlist(lapply(files, read.csv, header=TRUE), fill=TRUE))

  return(frame)
}

# --------- ANALYSIS PIPELINE --------- #
process_runs <- function(runs, pattern, between_vars, within_vars) {
  # process the specified runs sequentially.
  d <- NULL

  # read data from current directory and bind it to the rest.
  for (i in seq_along(runs)) {
    temp <- read_data(runs[i], pattern=pattern)
    temp$RunID <- i

    d <- rbind(d, temp)
  }

  # ensure that the IVs and DVs are factors.
  factorvars <- vapply(d[, c(between_vars, within_vars), drop=FALSE], FUN=is.factor, FUN.VALUE=logical(1))

  if (!all(factorvars)) {
    nonfactorvars <- names(factorvars)[!factorvars]
    message("Automatically converting the following non-factors to factors: ",
            paste(nonfactorvars, collapse = ", "))
    d[nonfactorvars] <- lapply(d[nonfactorvars], factor)
  }

  return(d)
}

group_plot <- function(d, dep_var, within_vars, between_vars, id_var=NULL, runs=0, run_names=NULL, meta_dir=NULL,
                       formula=NULL, title="", xlab="", ylab="", x_var=NULL, fill_var=NULL, facet_var=NULL,
                       group_var=NULL, pattern="*.csv", target_file="", yrange=NULL, line=F, palette="viridis",
                       all_base=NULL) {
  #' @title Group-level analysis and plotting
  #'
  #' @description Performs a statistical analysis of aggregated tabular simulation data from the specified runs,
  #' then saves the resulting plots in the `output` subdirectory.
  #'
  #' @param dep_var string: name of dependent variable (what is being measured).
  #' @param within_vars vector[string]: name(s) of within-participant independent variables.
  #' @param between_vars vector[string]: name(s) of between-participant independent variables.
  #' @param id_var string: name of variable identifying individual entries, e.g. participant or trial ID.
  #'
  #' @param runs vector[integer or string]: optional, lists run directories to be read, expressed in terms of
  #' timestamp recency or directory names. By default, processes most recent timestamped run directory.
  #' @param run_names vector[string]: optional, name for analysis output directories corresponding to each run.
  #'
  #' @param formula string: formula by which to aggregate the tabular data, e.g. Accuracy ~ Batch*Shots*Trial.
  #' Variables that are not to be plotted should not be included in the formula s.t. they are averaged over.
  #'
  #' @param title string: plot title.
  #' @param xlab string: x-axis label.
  #' @param ylab string: y-axis label.
  #'
  #' @param x_var string: name of variable to be plotted in x-axis.
  #' @param fill_var string: name of variable whose levels will be used to color the bars and dots.
  #' @param facet_var string: name of variable whose levels will be plotted as facets (distinct sub-panels).
  #'
  #' @examples
  #' group_plot(dep_var="Speed", within_vars="Car", between_vars="Team", id_var="Subject", runs=0,
  #'            run_names="test", formula="Speed~Car*Team*Subject",
  #'            title="Speed by Car X Team", xlab="Car", ylab="Speed",
  #'            x_var="Car", fill_var="Car", facet_var="Team")

  # generate run-wise or group-level plots, depending on `meta_dir`.
  for (i in seq_along(runs)) {
    # set output path for this run.
    if (!is.null(meta_dir)) {
      df <- d
      output_path <- file.path("analysis", meta_dir)
    } else {
      df <- d[d$RunID==i,]
      if (is.null(names)) {
        output_path <- file.path("results", runs[i])
      } else {
        output_path <- file.path("results", run_names[i])
      }
    }

    # create output directory for this run.
    dir.create(output_path, showWarnings=FALSE, recursive=TRUE)

    # aggregate over levels of the unplotted variables (to avoid element duplication in plot).
    df <- aggregate(formula(formula), df, mean)

    # generate plot.
    func = ifelse(line, line_plot, bar_plot)
    ggsave(filename = file.path(output_path, target_file), width=24, height=12, units="in", device="svg", dpi=600,
           plot = func(df, dep_var, within_vars, between_vars, id_var, x_var=x_var, fill_var=fill_var,
                       facet_var=facet_var, group_var=group_var, title=title, xlab=xlab, ylab=ylab, yrange=yrange,
                       palette=palette, all_base=all_base))
  }
}

# --------- Regularization Analysis --------- #
regularization <- function(cond_inclusion, layer_inclusion, in_pattern, out_fid, agg_fun,
                           line=F, incl_class=F, contr_dup=T, x_var="Duplication") {
  if(incl_class) {
    frml <- "Utilization ~ Analysis*Session*Condition*Variant*Smoothing*Layer*Class*Duplication*Resolution*Fold"
    fill_var <- "Class"
    within_vars <- c("Duplication", "Resolution", "Smoothing", "Layer", "Class")
  } else {
    frml <- "Utilization ~ Analysis*Session*Condition*Variant*Smoothing*Layer*Duplication*Resolution*Fold"
    fill_var <- "Condition"
    within_vars <- c("Duplication", "Resolution", "Smoothing", "Layer")
  }
  if (line) {
    within_vars <- append(within_vars, "Fold")
  }
  between_vars <- c("Session", "Condition")

  id_var <- "Analysis"
  dep_var <- "Utilization"

  if (length(cond_inclusion) > 1) {
    facet_var <- "Session"
  } else if (length(layer_inclusion) > 1) {
    facet_var <- "Layer"
  } else {
    facet_var <- NULL
  }

  # patch for final GLM case, combining homogeneous and uniform for duplication-wise comparisons.
  emm_options(pbkrtest.limit=15000)
  emm_options(lmerTest.limit=15000)

  # analyze and plot utilization data.
  util <- process_runs(run_dirs, in_pattern, between_vars, within_vars)
  util <- util[util$Smoothing %in% smoothing_levels,]

  util$Condition <- factor(util$Condition, levels=c("Baseline", "Homogeneous", "Uniform", "Scaled", "Adaptive"))
  util$Condition[util$Smoothing==.5] <- factor("Scaled")
  util$Condition[util$Condition=="Baseline"] <- factor("Homogeneous")

  util <- util[util$Layer %in% layer_inclusion,]
  util <- util[util$Condition %in% cond_inclusion,]

  util$Variant <- factor(util$Variant)
  util$HomWeight <- util$Variant

  if ("Homogeneous" %in% cond_inclusion) {
      gain_levels <- list("25"=c("con_b1", "sat_b1"), "28"=c("con_b2", "sat_b2"), "32"=c("con_b3", "sat_b3"),
                      "37"=c("con_b4", "sat_b4"), "45"=c("con_b5", "sat_b5"), "56"=c("con_b6", "sat_b6"),
                      "73"=c("con_b7", "sat_b7"), "109"=c("con_b8", "sat_b8"), "208"=c("con_b9", "sat_b9"),
                      "2500"=c("con_b10", "sat_b10"))
      levels(util$HomWeight) <- gain_levels

      util$HomWeight <- factor(util$HomWeight, levels=c("25", "28", "32", "37", "45", "56", "73", "109", "208", "2500", "0"))
      util$HomWeight[is.na(util$HomWeight)] <- 0
  }
  util <- util[util$Duplication != 1, ]
  frml2 <- paste(frml, "*HomWeight", sep="")

  # if (!line) {
    # ignore homogeneous duplication factors when aggregating.
  #  frml <- gsub("\\*Variant", "", frml)
  # }

  # compute function (sd) WITHIN levels of HomWeight first.
  std <- aggregate(formula(frml2), util, agg_fun)

  if (!line) {
    frml <- gsub("\\*Variant", "", frml)
    std <- aggregate(formula(frml), std, mean)
  } else {
    std <- aggregate(formula(frml2), std, mean)
    within_vars <- append(within_vars, "HomWeight")
  }

  # if (!line) {
    # aggregate over homogeneous duplication factors (don't exist in the original formula).
    # this computes std ACROSS levels of HomWeight, potentially inflating it for homogeneous.
  #  frml <- gsub("\\*Variant", "", frml)
  #  std <- aggregate(formula(frml), util, agg_fun)
  #} else {
    # retain homogeneous duplication factor levels.
    # this computes std WITHIN levels of HomWeight, the correct thing to do.
    # ONLY THEN should we average over those levels if we don't care about them.
  #  std <- aggregate(formula(frml), util, agg_fun)
  #  within_vars <- append(within_vars, "HomWeight")
  # }

  contr_type <- ifelse(contr_dup, "consec", "pairwise")

  if (!is.null(meta_dir)) {
    if(length(cond_inclusion) == 1) {
      std_glm <- lmer(Utilization ~ Duplication*Session + (1|Fold), std)
      std_omnibus <- joint_tests(std_glm, lmer.df="kenward-roger")
      std_contr <- contrast(emmeans(std_glm, ~ Duplication), contr_type)
      descriptive <- summaryBy(Utilization ~ Duplication, data = std,
                         FUN = function(x) {c(m = mean(x), s = sd(x))})
    } else {
      if (length(layer_inclusion) == 1) {
        std_glm <- lmer(Utilization ~ Condition*Duplication*Session + (1|Fold), std)
        std_omnibus <- joint_tests(std_glm, lmer.df="kenward-roger")
        if (contr_dup) {
          std_contr <- contrast(emmeans(std_glm, ~ Duplication|Condition), contr_type)
          descriptive <- summaryBy(Utilization ~ Duplication|Condition, data = std,
                                   FUN = function(x) {c(m = mean(x), s = sd(x))})
        } else {
          std_contr <- contrast(emmeans(std_glm, ~ Condition|Duplication), contr_type)
          descriptive <- summaryBy(Utilization ~ Condition|Duplication, data = std,
                                   FUN = function(x) {c(m = mean(x), s = sd(x))})
        }
      } else {
        std_glm <- lmer(Utilization ~ Layer*Condition*Duplication*Session + (1|Fold), std)
        std_omnibus <- joint_tests(std_glm, lmer.df="kenward-roger")
        if (contr_dup) {
          std_contr <- contrast(emmeans(std_glm, ~ Duplication|Condition|Layer), contr_type)
          descriptive <- summaryBy(Utilization ~ Duplication|Condition|Layer, data = std,
                                   FUN = function(x) {c(m = mean(x), s = sd(x))})
        } else {
          std_contr <- contrast(emmeans(std_glm, ~ Condition|Duplication|Layer), contr_type)
          descriptive <- summaryBy(Utilization ~ Layer|Session|Condition|Duplication, data = std,
                                   FUN = function(x) {c(m = mean(x), s = sd(x))})
        }
      }
    }
  }

  yrng <- c(0, max(std$Utilization))
  # yrng <- NULL
  group_plot(d=std, dep_var=dep_var, within_vars=within_vars, between_vars=between_vars, id_var=id_var,
             runs=run_dirs, run_names=run_names, meta_dir=meta_dir, formula=ifelse(!line, frml, frml2),
             title=sprintf("%s Utilization", out_fid), xlab=x_var, ylab=dep_var, yrange=yrng,
             x_var=x_var, fill_var=fill_var, facet_var=facet_var, group_var=group_var, pattern="util_(.*).csv",
             target_file=sprintf("util_%s.svg", out_fid), line=line,
             palette=ifelse(incl_class, "Spectral", "viridis"))

  return(list(std_omnibus, std_contr, descriptive))
}

# --------- Discrimination Analysis --------- #
discrimination <- function(cond_inclusion, layer_inclusion, in_pattern, out_fid, line=F, contr_dup=T, x_var="Duplication") {
  id_var <- "Analysis"
  dep_var <- "Accuracy"

  if (line) {
    within_vars <- c("Shots", "Duplication", "Resolution", "Layer", "Fold")
  } else {
    within_vars <- c("Shots", "Duplication", "Resolution", "Layer")
  }
  between_vars <- c("Session", "Condition")

  fill_var <- ifelse(length(cond_inclusion)==1, "Layer", "Condition")
  facet_var <- "Session"

  frml <- "Accuracy ~ Analysis*Session*Condition*Variant*Shots*Layer*Duplication*Resolution*Fold*RunID"

  # analyze and plot discrimination performance data.
  clf <- process_runs(run_dirs, in_pattern, between_vars, within_vars)
  clf <- clf[clf$Smoothing %in% smoothing_levels,]

  clf$Condition <- factor(clf$Condition, levels=c("Baseline", "Homogeneous", "Uniform", "Scaled", "Adaptive"))
  clf$Condition[clf$Smoothing==.5] <- factor("Scaled")
  clf$Condition[clf$Condition=="Baseline"] <- factor("Homogeneous")

  gain_levels <- list("25"=c("con_b1", "sat_b1"), "28"=c("con_b2", "sat_b2"), "32"=c("con_b3", "sat_b3"),
                "37"=c("con_b4", "sat_b4"), "45"=c("con_b5", "sat_b5"), "56"=c("con_b6", "sat_b6"),
                "73"=c("con_b7", "sat_b7"), "109"=c("con_b8", "sat_b8"), "208"=c("con_b9", "sat_b9"),
                "2500"=c("con_b10", "sat_b10"))

  clf <- clf[clf$Layer %in% layer_inclusion,]

  all_base <- clf[clf$Variant %in% unlist(gain_levels, use.names=F),]
  all_base <- aggregate(Accuracy ~ Variant*Fold, all_base, mean)

  clf <- clf[clf$Condition %in% cond_inclusion,]

  clf$HomWeight <- factor(clf$Variant)
  if ("Homogeneous" %in% cond_inclusion) {
      levels(clf$HomWeight) <- gain_levels

      clf$HomWeight <- factor(clf$HomWeight, levels=c("25", "28", "32", "37", "45", "56", "73", "109", "208", "2500", "0"))
      clf$HomWeight[is.na(clf$HomWeight)] <- 0
  }
  clf <- clf[clf$Duplication != 1, ]
  frml2 <- paste(frml, "*HomWeight", sep="")

  if (!line) {
    # aggregate over homogeneous duplication factors (don't exist in the original formula).
    frml <- gsub("\\*Variant", "", frml)
    clf <- aggregate(formula(frml), clf, mean)
  } else {
    # retain homogeneous duplication factor levels.
    clf <- aggregate(formula(frml2), clf, mean)
    within_vars <- append(within_vars, "HomWeight")
  }

  contr_type <- ifelse(contr_dup, "consec", "pairwise")

  if (!is.null(meta_dir)) {
    if(length(cond_inclusion) == 1) {
      clf_glm <- lmer(Accuracy ~ Duplication*Session + (1|Fold), clf)
      clf_omnibus <- joint_tests(clf_glm, lmer.df="kenward-roger")
      clf_contr <- contrast(emmeans(clf_glm, ~ Duplication), contr_type)
    } else {
      if (length(layer_inclusion) == 1) {
        clf_glm <- lmer(Accuracy ~ Condition*Duplication*Session + (1|Fold), clf)
        clf_omnibus <- joint_tests(clf_glm, lmer.df="kenward-roger")
        if (contr_dup) {
          clf_contr <- contrast(emmeans(clf_glm, ~ Duplication|Condition), contr_type)
        } else {
          clf_contr <- contrast(emmeans(clf_glm, ~ Condition|Duplication), contr_type)
        }
      } else {
        clf_glm <- lmer(Accuracy ~ Layer*Condition*Duplication*Session + (1|Fold), clf)
        clf_omnibus <- joint_tests(clf_glm, lmer.df="kenward-roger")
        if (contr_dup) {
          clf_contr <- contrast(emmeans(clf_glm, ~ Duplication|Condition|Layer), contr_type)
        } else {
          clf_contr <- contrast(emmeans(clf_glm, ~ Condition|Duplication|Layer), contr_type)
        }
      }
    }
  }

  yrng <- ifelse(("Homogeneous" %in% cond_inclusion) & length(cond_inclusion) == 1, c(0, 109), c(50, 109))
  # yrng <- NULL
  d <- group_plot(d=clf, dep_var=dep_var,
                  within_vars=within_vars, between_vars=between_vars, id_var=id_var,
                  runs=run_dirs, run_names=run_names, meta_dir=meta_dir,
                  formula=ifelse(!line, frml, frml2),
                  title=sprintf("%s Classification Accuracy", out_fid),
                  xlab=x_var, ylab=dep_var, yrange=yrng,
                  x_var=x_var, fill_var=fill_var, facet_var=facet_var, pattern="sep_svm(.*)_group.csv",
                  target_file=sprintf("clf_%s.svg", out_fid), line=line, all_base=NULL)

  if (length(layer_inclusion) == 1) {
    descriptive <- summaryBy(Accuracy ~ Session|Condition|Duplication, data = clf,
                             FUN = function(x) {c(m = mean(x), s = sd(x))})
  } else {
    descriptive <- summaryBy(Accuracy ~ Layer|Session|Condition|Duplication, data = clf,
                             FUN = function(x) {c(m = mean(x), s = sd(x))})
  }

  return(list(clf_omnibus, clf_contr, descriptive))
}

# --------- Regularization Within Condition (Class-specific line plots) --------- #
util_base_m <- regularization(c("Homogeneous"), c("MC", "GC"), "util_(.*).csv", "Base_M", mean, T, T, F, "HomWeight")
util_base_sd <- regularization(c("Homogeneous"), c("MC", "GC"), "util_(.*).csv", "Base_SD", sd, T, T, F, "HomWeight")

util_uniform_m <- regularization(c("Uniform"), c("MC", "GC"), "util_(.*).csv", "Uniform_M", mean, T, T, F)
util_uniform_sd <- regularization(c("Uniform"), c("MC", "GC"), "util_(.*).csv", "Uniform_SD", sd, T, T, F)

util_scaled_m <- regularization(c("Scaled"), c("MC", "GC"), "util_(.*).csv", "Scaled_M", mean, T, T, F)
util_scaled_sd <- regularization(c("Scaled"), c("MC", "GC"), "util_(.*).csv", "Scaled_SD", sd, T, T, F)

util_adaptive_m <- regularization(c("Adaptive"), c("MC", "GC"), "util_(.*).csv", "Adaptive_M", mean, T, T, F)
util_adaptive_sd <- regularization(c("Adaptive"), c("MC", "GC"), "util_(.*).csv", "Adaptive_SD", sd, T, T, F)

# --------- Regularization Across Conditions (Line and bar plots) --------- #
util_mc_m <- regularization(c("Adaptive", "Scaled", "Uniform", "Homogeneous"), c("MC"), "util_MC.csv", "MC_bar_M", mean, F, F, F)
util_gc_m <- regularization(c("Adaptive", "Scaled", "Uniform", "Homogeneous"), c("GC"), "util_GC.csv", "GC_bar_M", mean, F, F, F)

util_mc_m <- regularization(c("Adaptive", "Scaled", "Uniform", "Homogeneous"), c("MC"), "util_MC.csv", "MC_line_M", mean, T, F, F)
util_gc_m <- regularization(c("Adaptive", "Scaled", "Uniform", "Homogeneous"), c("GC"), "util_GC.csv", "GC_line_M", mean, T, F, F)

util_mc_sd <- regularization(c("Adaptive", "Scaled", "Uniform", "Homogeneous"), c("MC"), "util_MC.csv", "MC_bar_SD", sd, F, F, F)
util_gc_sd <- regularization(c("Adaptive", "Scaled", "Uniform", "Homogeneous"), c("GC"), "util_GC.csv", "GC_bar_SD", sd, F, F, F)

util_mc_sd <- regularization(c("Adaptive", "Scaled", "Uniform", "Homogeneous"), c("MC"), "util_MC.csv", "MC_line_SD", sd, T, F, F)
util_gc_sd <- regularization(c("Adaptive", "Scaled", "Uniform", "Homogeneous"), c("GC"), "util_GC.csv", "GC_line_SD", sd, T, F, F)

util_comb <- regularization(c("Uniform", "Homogeneous"), c("MC", "GC"), "util_(.*).csv", "MCGC_line_SD", sd, F, F, F)
util_comb_dup <- regularization(c("Uniform", "Homogeneous"), c("MC", "GC"), "util_(.*).csv", "MCGC_line_SD", sd, F, F, T)
util_full <- regularization(c("Adaptive", "Scaled", "Uniform", "Homogeneous"), c("MC", "GC"), "util_(.*).csv", "MCGC_line_SD", sd, F, F, F)
util_full_dup <- regularization(c("Adaptive", "Scaled", "Uniform", "Homogeneous"), c("MC", "GC"), "util_(.*).csv", "MCGC_line_SD", sd, F, F, T)


# --------- Intra-Condition Classification --------- #
clf_base <- discrimination(c("Homogeneous"), c("ET", "MC", "GC"), "sep_svm(.*)_group.csv", "Base", T, F, "HomWeight")
clf_uniform <- discrimination(c("Uniform"), c("ET", "MC", "GC"), "sep_svm(.*)_group.csv", "Uniform", T,F)
clf_scaled <- discrimination(c("Scaled"), c("ET", "MC", "GC"), "sep_svm(.*)_group.csv", "Scaled", T, F)
clf_adaptive <- discrimination(c("Adaptive"), c("ET", "MC", "GC"), "sep_svm(.*)_group.csv", "Adaptive", T, F)

clf_et <- discrimination(c("Adaptive", "Scaled", "Uniform", "Homogeneous"), c("ET"), "sep_svm(.*)_group.csv", "ET_bar", F, F)
clf_mc <- discrimination(c("Adaptive", "Scaled", "Uniform", "Homogeneous"), c("MC"), "sep_svm(.*)_group.csv", "MC_bar", F, F)
clf_gc <- discrimination(c("Adaptive", "Scaled", "Uniform", "Homogeneous"), c("GC"), "sep_svm(.*)_group.csv", "GC_bar", F, F)

clf_mc <- discrimination(c("Adaptive", "Scaled", "Uniform", "Homogeneous"), c("MC"), "sep_svm(.*)_group.csv", "MC_line", T, F)
clf_gc <- discrimination(c("Adaptive", "Scaled", "Uniform", "Homogeneous"), c("GC"), "sep_svm(.*)_group.csv", "GC_line", T, F)

clf_comb <- discrimination(c("Uniform", "Homogeneous"), c("MC", "GC"), "sep_svm(.*)_group.csv", "MCGC_bar", F, F)
clf_comb_dup <- discrimination(c("Uniform", "Homogeneous"), c("MC", "GC"), "sep_svm(.*)_group.csv", "MCGC_bar", F, T)
clf_full <- discrimination(c("Adaptive", "Scaled", "Uniform", "Homogeneous"), c("MC", "GC"), "sep_svm(.*)_group.csv", "MCGC_bar", F, F)
clf_full_dup <- discrimination(c("Adaptive", "Scaled", "Uniform", "Homogeneous"), c("MC", "GC"), "sep_svm(.*)_group.csv", "MCGC_bar", F, T)

# --------- Log Results --------- #
if (!dir.exists(file.path("analysis", meta_dir, "csv"))) {
  dir.create(file.path("analysis", meta_dir, "csv"))
}

# dump console output to this file #
if (!is.null(meta_dir)) {
  sink(file=file.path("analysis", meta_dir, "analysis_base.txt"))

  cat(paste("\nDiscrimination Accuracy\n******************\n"))
  for (item in clf_base) {
    print(item)
    cat("\n")
  }

  cat(paste("\nMean Utilization\n**************\n"))
  for (item in util_base_m) {
    print(item)
    cat("\n")
  }

  cat(paste("\nSD Utilization\n**************\n"))
  for (item in util_base_sd) {
    print(item)
    cat("\n")
  }

  sink(NULL)
  sink(file=file.path("analysis", meta_dir, "analysis_exp.txt"))

  cat(paste("\nFigure 3-4 (Het-Hom Utilization by Condition)\n**************************\n"))
  for (i in seq_along(util_comb)) {
    print(util_comb[i])
    cat("\n")
    write.table(util_comb[i], file.path("analysis", meta_dir, "csv", paste("hh_util_cond_", i, ".csv", sep="")),
                row.names=FALSE, quote=FALSE, sep=',')
  }

  cat(paste("\nFigure 4C (Het-Hom Utilization by Duplication)\n**************************\n"))
  for (i in seq_along(util_comb_dup)) {
    print(util_comb_dup[i])
    cat("\n")
    write.table(util_comb_dup[i], file.path("analysis", meta_dir, "csv", paste("hh_util_dup_", i, ".csv", sep="")),
            row.names=FALSE, quote=FALSE, sep=',')
  }

  cat(paste("\nFigure 3-4 (Het-Hom Accuracy by Condition)\n**************************\n"))
  for (i in seq_along(clf_comb)) {
    print(clf_comb[i])
    cat("\n")
    write.table(clf_comb[i], file.path("analysis", meta_dir, "csv", paste("hh_clf_cond_", i, ".csv", sep="")),
        row.names=FALSE, quote=FALSE, sep=',')
  }

  cat(paste("\nFigure 3-4 (Het-Hom Accuracy by Duplication)\n**************************\n"))
  for (i in seq_along(clf_comb_dup)) {
    print(clf_comb_dup[i])
    cat("\n")
    write.table(clf_comb_dup[i], file.path("analysis", meta_dir, "csv", paste("hh_clf_dup_", i, ".csv", sep="")),
                row.names=FALSE, quote=FALSE, sep=',')
  }

  cat(paste("\nFigure 5 (Combined Utilization by Condition)\n**************************\n"))
  for (i in seq_along(util_full)) {
    print(util_full[i])
    cat("\n")
    write.table(util_full[i], file.path("analysis", meta_dir, "csv", paste("full_util_cond_", i, ".csv", sep="")),
            row.names=FALSE, quote=FALSE, sep=',')
  }

  cat(paste("\nFigure 5 (Combined Utilization by Duplication)\n**************************\n"))
  for (i in seq_along(util_full_dup)) {
    print(util_full_dup[i])
    cat("\n")
    write.table(util_full_dup[i], file.path("analysis", meta_dir, "csv", paste("full_util_dup_", i, ".csv", sep="")),
        row.names=FALSE, quote=FALSE, sep=',')
  }

  cat(paste("\nFigure 5 (Combined Accuracy by Condition)\n**************************\n"))
  for (i in seq_along(clf_full)) {
    print(clf_full[i])
    cat("\n")
    write.table(clf_full[i], file.path("analysis", meta_dir, "csv", paste("full_clf_cond_", i, ".csv", sep="")),
                row.names=FALSE, quote=FALSE, sep=',')
  }

  cat(paste("\nFigure 5 (Combined Accuracy by Duplication)\n**************************\n"))
  for (i in seq_along(clf_full_dup)) {
    print(clf_full_dup[i])
    cat("\n")
    write.table(clf_full_dup[i], file.path("analysis", meta_dir, "csv", paste("full_clf_dup_", i, ".csv", sep="")),
            row.names=FALSE, quote=FALSE, sep=',')
  }

  cat(paste("\nET Discrimination Accuracy\n**************************\n"))
  for (item in clf_et) {
    print(item)
    cat("\n")
  }

  cat(paste("\nMC Discrimination Accuracy\n**************************\n"))
  for (item in clf_mc) {
    print(item)
    cat("\n")
  }

  cat(paste("\nGC Discrimination Accuracy\n**************************\n"))
  for (item in clf_gc) {
    print(item)
    cat("\n")
  }

  cat(paste("\nMC SD Utilization\n*****************\n"))
  for (item in util_mc_sd) {
    print(item)
    cat("\n")
  }

  cat(paste("\nGC SD Utilization\n*****************\n"))
  for (item in util_gc_sd) {
    print(item)
    cat("\n")
  }

  cat(paste("\nMC Mean Utilization\n*****************\n"))
  for (item in util_mc_m) {
    print(item)
    cat("\n")
  }

  cat(paste("\nGC Mean Utilization\n*****************\n"))
  for (item in util_gc_m) {
    print(item)
    cat("\n")
  }

  sink(NULL)
}
