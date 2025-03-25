# -------------- IMPORTS -------------- #
suppressPackageStartupMessages({
  library(ggplot2)
  library(ggbeeswarm)
  library(ggpp)
  library(vioplot)
  library(scales)
  library(see)
  library(plotrix)
})

# ------------- HELPERS ------------- #
position_nudgedodge <- function(x = 0, y = 0, width = 0.75) {
    ggproto(NULL, PositionNudgedodge,
        x = x,
        y = y,
        width = width
    )
}

PositionNudgedodge <- ggproto("PositionNudgedodge", PositionDodge,
    x = 0,
    y = 0,
    width = 0.3,

    setup_params = function(self, data) {
        l <- ggproto_parent(PositionDodge,self)$setup_params(data)
        append(l, list(x = self$x, y = self$y))
    },

    compute_layer = function(self, data, params, layout) {
        d <- ggproto_parent(PositionNudge,self)$compute_layer(data,params,layout)
        d <- ggproto_parent(PositionDodge,self)$compute_layer(d,params,layout)
        d
    }
)

#' Nudge any other positioning function from ggplot2
#' @param x Nudge in X direction
#' @param y Nudge in Y direction
#' @param position Any positioning operator from ggplot2 like \link{position_jitter}[ggplot2]
#' @return A combination positioning operator which first runs the original positioning,
#' then adds a nudge on top of that.
#' @export
position_nudge_any <- function(x = 0, y = 0, position) {
  ggproto(NULL, PositionNudgeAny,
          nudge = ggplot2::position_nudge(x, y),
          position = position
  )
}

#' Internal class doing the actual nudging on top of the other operation
#' @keywords internal
PositionNudgeAny <- ggplot2::ggproto("PositionNudgeAny", ggplot2::Position,
  nudge = NULL,
  nudge_params = NULL,
  position = NULL,
  position_params = NULL,

  setup_params = function(self, data) {
   list(nudge = self$nudge,
        nudge_params = self$nudge$setup_params(data),
        position = self$position,
        position_params = self$position$setup_params(data))
  },

  setup_data = function(self, data, params) {
   data <- params$position$setup_data(data, params$position_params)
   params$nudge$setup_data(data, params$nudge_params)
  },

  compute_layer = function(self, data, params, layout) {
   data <- params$position$compute_layer(data, params$position_params, layout)
   params$nudge$compute_layer(data, params$nudge_params, layout)
  }
)

pretty_lim <- function(x, n=5) {
  r2 <- ifelse(x<0, 0, x)
  pr <- pretty(r2,n)
  r_out <- range(pr)
  r_out
}


pretty_unexpanded <- function(x, n=5) {
  if(x[1]<=0){
    r2 <- x + c(-x[1],x[1])
  }else{
    r2 <- x + c((x[2]-x[1])*0.04545455,-(x[2]-x[1])*0.04545455)
  }
  pout <-  pretty(r2,n)
  pout
}

# --------- PLOTTING FUNCTIONS --------- #
publication_theme <- theme_bw() +
  theme(
    plot.title = element_text(size = 32, color="black", face = "bold"),
    axis.title = element_text(size = 24, color="black"),
    # axis.line = element_line(linewidth = 1),
    axis.text = element_text(size = 32, color="black"),
    axis.title.x = element_text(margin=margin(10,0,0,0)),
    axis.title.y = element_text(margin=margin(0,10,0,0)),
    strip.text = element_text(size = 24),
    strip.background = element_blank(),
    # legend.position = c(.99, .99),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(6, 6, 6, 6),
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 18)
  )

theme2 <- theme(panel.background = element_rect(fill = 'white'),
                axis.text = element_text(size = 16, color="black"),
                plot.title = element_text(size = 16, face = "bold"),
                axis.title = element_text(size = 16, color = "black"),
                axis.title.x = element_text(margin=margin(10,0,0,0)),
                axis.title.y = element_text(margin=margin(0,10,0,0)),
                strip.text = element_text(size = 12),
                legend.justification = "right")

line_plot <- function(df, dep_var, within_vars, between_vars, id_var,
                     x_var=NULL, fill_var=NULL, facet_var=NULL, group_var=NULL,
                     title="", xlab="", ylab="", yrange=NULL, palette="viridis", all_base=NULL) {
  #' @title Multiline plot with datapoints and shaded CI ribbons.
  #' @description Plots the requested variables after computing full descriptive statistics.
  if (!is.null(within_vars)) {
    descriptives <- summarySEwithin(df, measurevar=dep_var, withinvars=within_vars,
                                    betweenvars=between_vars, idvar=id_var)
  } else {
    descriptives <- summarySE(df, measurevar=dep_var, groupvars=between_vars)
  }

  p <- ggplot(descriptives, aes(x=.data[[x_var]], y=.data[[dep_var]], fill=.data[[fill_var]], group=.data[[fill_var]]))

  if (!is.null(facet_var)) {
    p <- p + facet_wrap(~.data[[facet_var]])
  }

  p <- p + stat_summary(fun = mean, geom = "line", aes(color=.data[[fill_var]], group=.data[[fill_var]]), linewidth=1.25)
  p <- p + stat_summary(fun = mean, geom = "point", aes(group=.data[[fill_var]]))
  p <- p + stat_summary(geom="ribbon", fun.data = mean_se, aes(group=.data[[fill_var]]), alpha=.1)

  if (palette == "viridis") {
    p <- p + scale_fill_viridis_d() + scale_color_viridis_d()
  } else {
    p <- p + scale_fill_brewer(palette=palette) + scale_color_brewer(palette=palette)
  }
  p <- p + theme(plot.title = element_text(hjust = 0.5)) + publication_theme

  if (!is.null(yrange)) {
    p <- p + coord_cartesian(ylim=c(yrange[1], yrange[2]))
  } else {
    p <- p + coord_cartesian(ylim=c(max(min(df[dep_var]-descriptives$ci), 0), max(df[dep_var]+descriptives$ci)))
  }

  p <- p + scale_y_continuous(breaks = breaks_pretty(5))

}

bar_plot <- function(df, dep_var, within_vars, between_vars, id_var,
                     x_var=NULL, fill_var=NULL, facet_var=NULL, group_var=NULL,
                     title="", xlab="", ylab="", yrange=NULL, beeswarm=T, palette="viridis",
                     all_base=NULL) {
  #' @title Bar plot with dots and CI.
  #' @description Plots the requested variables after computing full descriptive statistics.
  #' Levels of 'x_var' will be plotted as bars, levels of 'facet_var' as facets.
  #' Title and axis labels may also be provided.
  if (!is.null(within_vars)) {
    descriptives <- summarySEwithin(df, measurevar=dep_var, withinvars=within_vars,
                                    betweenvars=between_vars, idvar=id_var)
  } else {
    descriptives <- summarySE(df, measurevar=dep_var, groupvars=between_vars)
  }

  # determine how each variable will be plotted.
  vars <- c(within_vars, between_vars)

  x_var <- ifelse(is.null(x_var), vars[1], x_var)
  fill_var <- ifelse(is.null(fill_var), vars[ifelse(length(var)>1, 2, 1)], fill_var)
  # facet_var <- ifelse(is.null(facet_var), vars[ifelse(length(vars)>2, 3, 2)], facet_var)

  # plotting control.
  bar_width <- 1.0
  bar_pad <- bar_width/2.0

  violin_width <- bar_width/1.5
  line_width <- 0.3
  err_width <- 0.02

  if (x_var != fill_var) {
    swarm_nudge_x <- -bar_width/(2 * length(unique(df[[fill_var]])))
  } else {
    swarm_nudge_x <- -bar_width/1.5
  }
  ci_nudge_x <- swarm_nudge_x + err_width/1.5

  # compute external baseline while respecting levels of facet variable.
  if (!is.null(all_base)) {
    all_base$Session <- ifelse(grepl("con", all_base$Variant), "Concentration", "Saturation")
    all_base <- aggregate(formula(paste(dep_var, "~ Session")), all_base, function(x) c(mean = mean(x), se = std.error(x)))
    all_base <- do.call(data.frame, all_base)

    descriptives$base_m <- ifelse(descriptives$Session=="Concentration",
                                  all_base[all_base$Session=="Concentration",]$Accuracy.mean,
                                  all_base[all_base$Session=="Saturation",]$Accuracy.mean)

    descriptives$base_se <- ifelse(descriptives$Session=="Concentration",
                              all_base[all_base$Session=="Concentration",]$Accuracy.se,
                              all_base[all_base$Session=="Saturation",]$Accuracy.se)
  }

  # construct bar plot with data dots, SE wicks, and 95% CI ranges.
  p <- ggplot(descriptives, aes(x=.data[[x_var]], y=.data[[dep_var]], fill=.data[[fill_var]]))

  p <- p + geom_bar(position=position_dodge2(preserve="total", padding=bar_pad),
                    stat="identity", width=bar_width, alpha=1)

  if (!is.null(facet_var)) {
    p <- p + facet_wrap(~.data[[facet_var]])
  }

  # p <- p + geom_violin(data=df, aes(x=.data[[x_var]], y=.data[[dep_var]]), width=.6, adjust=2,
  #                     position = position_dodge(width=bar_width), alpha=.3)

  # custom baseline mean and SE ribbon.
  if (!is.null(all_base)) {
    p <- p + geom_hline(aes(yintercept = .data[["base_m"]]), alpha=1, color="darkgreen")
    p <- p + geom_rect(aes(xmin=-Inf, xmax=Inf, ymin=.data[["base_m"]]-.data[["base_se"]], ymax=.data[["base_m"]]+.data[["base_se"]]),
                         alpha=.01, color=NA, fill="darkgreen")
  }

  p <- p + geom_linerange(aes(ymin = .data[[dep_var]]-ci, ymax = .data[[dep_var]]+ci), linewidth=line_width, alpha=.7,
                          position = position_nudge_any(x=ci_nudge_x, y=0, position_dodge(width=bar_width)), color = "red")

  p <- p + ggtitle(title) + labs(x=xlab, y=ylab)
  p <- p + geom_errorbar(aes(ymin=.data[[dep_var]]-se, ymax=.data[[dep_var]]+se), width=err_width,
                           linewidth=line_width, position = position_dodge(width=bar_width), color='black')

  if (!beeswarm) {
    dot_position <- position_nudge(x = -.48)
  } else {
    dot_position <- position_nudge_any(x=swarm_nudge_x, y=0, position_beeswarm(dodge.width=bar_width, side=-1L, cex=.75))
  }

  p <- p + geom_point(
  data=df, aes(x=.data[[x_var]], y=.data[[dep_var]], color=.data[[fill_var]]), position=dot_position, size=.75)

  # set a reasonable y axis range (DV).
  if (!is.null(yrange)) {
    p <- p + coord_cartesian(ylim=c(yrange[1], yrange[2]))
  } else {
    p <- p + coord_cartesian(ylim=c(max(min(df[dep_var]-descriptives$ci), 0), max(df[dep_var]+descriptives$ci)))
  }

  p <- p + scale_y_continuous(breaks = breaks_pretty(5))
  p <- p + guides(fill=guide_legend(title=fill_var), color="none", override.aes = list(shape = NA))

  # p <- p + scale_fill_manual(values=c("#42be71", "#2a788e", "#FFA500", "#453781"), name=xlab) +
  #  scale_color_manual(values=c("#42be71", "#2a788e", "#FFA500", "#453781"), name=xlab) +
  #  theme(plot.title = element_text(hjust = 0.5)) + theme

  if (palette == "viridis") {
    p <- p + scale_fill_viridis_d() + scale_color_viridis_d()

  } else {
    p <- p + scale_color_brewer(palette=palette)
  }
  p <- p + theme(plot.title = element_text(hjust = 0.5)) + publication_theme

  return(p)
}
