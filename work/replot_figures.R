suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(grid)
})

root <- "D:/TT paper/1107Invariant Continuous-Time"
data_dir <- file.path(root, "results", "figures_r", "data")
out_dir <- file.path(root, "results", "figures_r")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

ink <- "#000000"
mid <- "#666666"
light <- "#E8E8E8"
dark <- "#303030"
accent <- "#2C6E9B"
accent_pale <- "#D6E6F0"
blue_dark <- "#1F4E79"
blue_light <- "#8CB8D3"
pink <- "#D66BA0"
pink_light <- "#F2B6CF"
cohort_colors <- c(
  "MIMIC-IV" = blue_dark,
  "MIMIC-III" = blue_light,
  "eICU expanded" = pink
)

theme_pub <- theme_classic(base_family = "serif", base_size = 10) +
  theme(
    text = element_text(colour = ink),
    axis.text = element_text(colour = ink),
    axis.title = element_text(colour = ink),
    plot.title = element_text(colour = ink, face = "bold"),
    plot.subtitle = element_text(colour = ink),
    strip.text = element_text(colour = ink, face = "bold"),
    legend.text = element_text(colour = ink),
    legend.title = element_text(colour = ink),
    panel.grid.major = element_line(colour = "#DDDDDD", linewidth = 0.35),
    panel.grid.minor = element_blank(),
    plot.background = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA)
  )

save_plot <- function(plot, stem, width = 9.2, height = 7.2) {
  ggsave(file.path(out_dir, paste0(stem, ".png")), plot,
         width = width, height = height, dpi = 320, bg = "white")
  ggsave(file.path(out_dir, paste0(stem, ".pdf")), plot,
         width = width, height = height, bg = "white")
}

flow_panel <- function(nodes, edges, xlim, ylim, panel = NULL) {
  if (!"fill_colour" %in% names(nodes)) {
    nodes$fill_colour <- light
  }
  if (!"text_colour" %in% names(nodes)) {
    nodes$text_colour <- ink
  }
  plot <- ggplot() +
    geom_segment(
      data = edges,
      aes(x = x, y = y, xend = xend, yend = yend),
      arrow = arrow(length = unit(0.13, "inches")),
      linewidth = 0.5,
      colour = mid
    ) +
    geom_rect(
      data = nodes,
      aes(xmin = x - width / 2, xmax = x + width / 2,
          ymin = y - height / 2, ymax = y + height / 2),
      fill = nodes$fill_colour, colour = dark, linewidth = 0.5
    ) +
    geom_text(
      data = nodes,
      aes(x = x, y = y, label = label),
      family = "serif", colour = nodes$text_colour,
      size = 2.8, lineheight = 0.92
    ) +
    coord_cartesian(xlim = xlim, ylim = ylim, clip = "off") +
    theme_void(base_family = "serif") +
    theme(plot.background = element_rect(fill = "white", colour = NA))
  if (!is.null(panel)) {
    plot <- plot + annotate(
      "text", x = xlim[1] + 0.05, y = ylim[2] - 0.05,
      label = panel, family = "serif", fontface = "bold",
      colour = ink, size = 5
    )
  }
  plot
}

figure1 <- local({
  a_nodes <- data.frame(
    x = c(2.0, 5.5, 9.0, 5.5, 5.5),
    y = c(5.5, 5.5, 5.5, 3.6, 1.8),
    width = c(2.0, 2.0, 2.0, 4.0, 3.8),
    height = c(1.0, 1.0, 1.0, 1.0, 1.0),
    label = c(
      "MIMIC-IV\n13,131 / 4,475",
      "MIMIC-III\n3,470 / 409",
      "eICU expanded\n5,913 / 642",
      "24-hour landmark\nAlive and >=6 observed channels",
      "Analytic cohorts\n13,131 | 3,470 | 5,913"
    ),
    fill_colour = c(blue_dark, blue_light, pink, light, "#F4F7FA"),
    text_colour = c("white", ink, ink, ink, ink)
  )
  a_edges <- data.frame(
    x = c(2.0, 5.5, 9.0, 5.5),
    y = c(5.0, 5.0, 5.0, 3.1),
    xend = c(4.5, 5.5, 6.5, 5.5),
    yend = c(4.1, 4.1, 4.1, 2.3)
  )
  p1 <- flow_panel(a_nodes, a_edges, c(0, 11), c(1.1, 6.2), "A")

  components <- data.frame(
    cohort = rep(c("MIMIC-IV", "MIMIC-III", "eICU expanded"), each = 3),
    component = rep(c("Acute MI", "Stroke", "Death"), times = 3),
    events = c(3214, 1079, 524, 219, 199, 13, 459, 119, 90)
  )
  p2 <- ggplot(components, aes(x = cohort, y = events, fill = component)) +
    geom_col(position = "dodge", colour = dark, linewidth = 0.3, width = 0.72) +
    scale_fill_manual(values = c(
      "Acute MI" = blue_dark,
      "Stroke" = blue_light,
      "Death" = pink
    )) +
    labs(x = NULL, y = "Recorded events", fill = NULL,
         title = "B  Outcome components") +
    theme_pub + theme(legend.position = "top")

  c_nodes <- data.frame(
    x = c(1.1, 3.2, 5.5, 7.8, 9.9),
    y = 3.0,
    width = c(1.45, 1.55, 1.75, 1.65, 1.55),
    height = 1.0,
    label = c(
      "20 channels\n7 organs",
      "Observation\nmasks and time",
      "Dynamic\ngraph",
      "Transformer\nencoder",
      "MACCE\nrisk"
    )
  )
  c_edges <- data.frame(
    x = c(1.83, 3.98, 6.38, 8.63),
    y = 3.0,
    xend = c(2.47, 4.62, 6.98, 9.12),
    yend = 3.0
  )
  p3 <- flow_panel(c_nodes, c_edges, c(0, 10.8), c(2.2, 3.8), "C")

  d_nodes <- data.frame(
    x = c(1.1, 3.3, 5.6, 7.8, 9.9),
    y = c(3.0, 3.0, 3.0, 3.0, 3.0),
    width = c(1.45, 1.55, 1.8, 1.65, 1.65),
    height = 1.0,
    label = c(
      "2008-2016\nfit",
      "2017-2019\nselect",
      "2020-2022\nlocked test",
      "MIMIC-III and\neICU transport",
      "Calibration,\nDCA, endpoints"
    )
  )
  d_edges <- data.frame(
    x = c(1.83, 4.08, 6.5, 8.63),
    y = 3.0,
    xend = c(2.57, 4.82, 7.0, 9.07),
    yend = 3.0
  )
  p4 <- flow_panel(d_nodes, d_edges, c(0, 10.8), c(2.2, 3.8), "D")

  (p1 | p2) / (p3 | p4) +
    plot_annotation(
      title = "Overall study roadmap",
      theme = theme(
        plot.title = element_text(
          family = "serif", colour = ink, face = "bold", hjust = 0.5
        ),
        plot.background = element_rect(fill = "white", colour = NA)
      )
    )
})
save_plot(figure1, "figure1_overall_roadmap", 10.5, 8.0)

figure2 <- local({
  risk <- read_csv(file.path(data_dir, "cohort_risk_profiles.csv"),
                   show_col_types = FALSE)
  p1 <- ggplot(risk, aes(x = cohort, y = reorder(risk_factor, percent),
                         fill = percent, colour = cohort)) +
    geom_tile(linewidth = 1.0) +
    geom_text(aes(label = sprintf("%.1f", percent)),
              family = "serif", colour = ink, size = 3.0) +
    scale_fill_gradient(low = "#F2F2F2", high = accent, na.value = "white") +
    scale_colour_manual(values = cohort_colors, guide = "none") +
    labs(x = NULL, y = NULL, fill = "Percent",
         title = "A  Cohort risk-factor profile") +
    theme_pub + theme(axis.text.x = element_text(angle = 20, hjust = 1))

  baseline <- risk %>%
    filter(cohort == "MIMIC-IV") %>%
    select(risk_factor, baseline = percent)
  differences <- risk %>%
    left_join(baseline, by = "risk_factor") %>%
    filter(cohort != "MIMIC-IV") %>%
    mutate(difference = percent - baseline)
  p2 <- ggplot(differences, aes(x = difference,
                                y = reorder(risk_factor, difference),
                                colour = cohort, shape = cohort)) +
    geom_vline(xintercept = 0, colour = mid, linewidth = 0.4) +
    geom_point(size = 2.6, position = position_dodge(width = 0.35)) +
    scale_colour_manual(values = cohort_colors[c("MIMIC-III", "eICU expanded")]) +
    labs(x = "Percentage-point difference from MIMIC-IV", y = NULL,
         colour = NULL, shape = NULL,
         title = "B  Relative comorbidity profile") +
    theme_pub + theme(legend.position = "top")
  p1 | p2 + plot_layout(widths = c(1, 1.1))
})
save_plot(figure2, "figure2_cohort_profiles", 10.5, 5.6)

figure3 <- local({
  coverage <- read_csv(file.path(data_dir, "observation_coverage.csv"),
                       show_col_types = FALSE)
  p1 <- ggplot(coverage, aes(x = cohort, y = reorder(channel, coverage),
                             fill = coverage, colour = cohort)) +
    geom_tile(linewidth = 1.0) +
    geom_text(aes(label = sprintf("%.0f", 100 * coverage)),
              family = "serif", colour = ink, size = 2.7) +
    scale_fill_gradient(low = "#F2F2F2", high = accent, na.value = "white") +
    scale_colour_manual(values = cohort_colors, guide = "none") +
    labs(x = NULL, y = NULL, fill = "Coverage",
         title = "A  Channel observation coverage, percent") +
    theme_pub + theme(axis.text.x = element_text(angle = 20, hjust = 1))

  organ_summary <- coverage %>%
    group_by(cohort, organ) %>%
    summarise(mean_coverage = mean(coverage, na.rm = TRUE), .groups = "drop")
  p2 <- ggplot(organ_summary, aes(x = organ, y = mean_coverage,
                                  fill = cohort)) +
    geom_col(position = "dodge", colour = dark, linewidth = 0.3, width = 0.72) +
    scale_fill_manual(values = cohort_colors) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(x = NULL, y = "Mean coverage", fill = NULL,
         title = "B  Coverage by organ system") +
    theme_pub +
    theme(axis.text.x = element_text(angle = 25, hjust = 1),
          legend.position = "top")
  p1 | p2
})
save_plot(figure3, "figure3_observation_coverage", 10.5, 6.2)

figure4 <- local({
  dev <- read_csv(file.path(data_dir, "development_performance.csv"),
                  show_col_types = FALSE) %>%
    filter(model %in% c(
      "xgb_static", "xgb_static_ts", "transformer",
      "ct_graph", "organ_graph", "temporal_ensemble"
    )) %>%
    mutate(label = recode(
      model,
      xgb_static = "Static XGBoost",
      xgb_static_ts = "XGB + temporal summaries",
      transformer = "Transformer",
      ct_graph = "Dynamic organ graph",
      organ_graph = "Seven-organ token graph",
      temporal_ensemble = "Temporal ensemble"
    ), plot_color = if_else(model == "temporal_ensemble", pink, blue_dark))
  p1 <- ggplot(dev, aes(x = auroc, y = reorder(label, auroc))) +
    geom_segment(aes(x = 0.60, xend = auroc, yend = label),
                 colour = "#C9C9C9", linewidth = 0.8) +
    geom_point(aes(colour = plot_color), size = 2.8) +
    scale_colour_identity() +
    geom_text(aes(label = sprintf("%.2f", auroc)), hjust = -0.25,
              family = "serif", colour = ink, size = 2.9) +
    scale_x_continuous(limits = c(0.60, 0.80)) +
    labs(x = "AUROC", y = NULL, title = "A  Development performance") +
    theme_pub

  validation <- read_csv(file.path(data_dir, "temporal_validation.csv"),
                         show_col_types = FALSE)
  strong <- read_csv(file.path(data_dir, "strong_holdout.csv"),
                     show_col_types = FALSE) %>%
    mutate(set = "Locked 2020-2022")
  validation <- validation %>%
    mutate(set = "Fixed temporal holdout") %>%
    select(model, auroc, set)
  p2data <- bind_rows(validation, strong)
  p2 <- ggplot(p2data, aes(x = reorder(model, auroc), y = auroc,
                           fill = set)) +
    geom_col(position = "dodge", colour = dark, linewidth = 0.3, width = 0.7) +
    scale_fill_manual(values = c(
      "Fixed temporal holdout" = blue_light,
      "Locked 2020-2022" = pink
    )) +
    coord_cartesian(ylim = c(0.68, 0.78)) +
    labs(x = NULL, y = "AUROC", fill = NULL,
         title = "B  Temporal validation") +
    theme_pub +
    theme(axis.text.x = element_text(angle = 22, hjust = 1),
          legend.position = "top")

  transport <- read_csv(file.path(data_dir, "transport_summary.csv"),
                        show_col_types = FALSE) %>%
    filter(model %in% c("xgb_static_ts", "transformer",
                        "temporal_ensemble", "enhanced_fusion")) %>%
    mutate(
      cohort = recode(cohort, mimic3 = "MIMIC-III",
                      eicu_anycad = "eICU expanded"),
      model = recode(
        model,
        xgb_static_ts = "XGB + TS",
        transformer = "Transformer",
        temporal_ensemble = "Temporal ensemble",
        enhanced_fusion = "Enhanced fusion"
      )
    )
  p3 <- ggplot(transport, aes(x = model, y = auroc_raw,
                              colour = cohort, shape = cohort)) +
    geom_point(size = 2.8, position = position_dodge(width = 0.35)) +
    scale_colour_manual(values = cohort_colors[c("MIMIC-III", "eICU expanded")]) +
    coord_cartesian(ylim = c(0.62, 0.72)) +
    labs(x = NULL, y = "AUROC", colour = NULL, shape = NULL,
         title = "C  Transport discrimination") +
    theme_pub +
    theme(axis.text.x = element_text(angle = 20, hjust = 1),
          legend.position = "top")

  development <- read_csv(file.path(data_dir, "development_performance.csv"),
                          show_col_types = FALSE) %>%
    filter(model %in% c("xgb_static", "xgb_static_ts", "transformer",
                        "ct_graph", "organ_graph", "temporal_ensemble")) %>%
    mutate(plot_color = if_else(model == "temporal_ensemble", pink, blue_dark))
  p4 <- ggplot(development, aes(x = brier, y = reorder(model, brier))) +
    geom_point(aes(colour = plot_color), size = 2.5) +
    scale_colour_identity() +
    geom_segment(aes(x = 0.17, xend = brier, yend = model),
                 colour = "#C9C9C9", linewidth = 0.7) +
    labs(x = "Brier score", y = NULL, title = "D  Probability error") +
    theme_pub
  (p1 | p2) / (p3 | p4)
})
save_plot(figure4, "figure4_performance", 10.5, 8.2)

figure6 <- local({
  calibration <- read_csv(file.path(data_dir, "calibration_curves.csv"),
                          show_col_types = FALSE) %>%
    mutate(cohort = recode(cohort, mimic3 = "MIMIC-III",
                           eicu_anycad = "eICU expanded"))
  reliability <- function(cohort_name, panel) {
    colors <- if (cohort_name == "eICU expanded") {
      c("Raw" = pink_light, "Recalibrated" = pink)
    } else {
      c("Raw" = blue_light, "Recalibrated" = blue_dark)
    }
    ggplot(filter(calibration, cohort == cohort_name),
           aes(x = predicted, y = observed, colour = source,
               linetype = source)) +
      geom_abline(slope = 1, intercept = 0, colour = mid, linewidth = 0.4) +
      geom_line(linewidth = 0.8) +
      geom_point(size = 2.0) +
      scale_colour_manual(values = colors) +
      coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
      labs(x = "Predicted risk", y = "Observed risk",
           colour = NULL, linetype = NULL,
           title = paste0(panel, "  ", cohort_name)) +
      theme_pub + theme(legend.position = "top")
  }
  p1 <- reliability("MIMIC-III", "A")
  p2 <- reliability("eICU expanded", "B")

  dca <- read_csv(file.path(data_dir, "decision_curves.csv"),
                  show_col_types = FALSE) %>%
    filter(cohort == "eicu_anycad")
  p3 <- ggplot(dca, aes(x = threshold, y = net_benefit, colour = source,
                        fill = source)) +
    geom_ribbon(aes(ymin = low, ymax = high), alpha = 0.13, colour = NA) +
    geom_line(linewidth = 0.9) +
    geom_hline(yintercept = 0, colour = ink, linewidth = 0.4) +
    scale_colour_manual(values = c("Raw" = pink_light, "Recalibrated" = pink)) +
    scale_fill_manual(values = c("Raw" = pink_light, "Recalibrated" = pink)) +
    labs(x = "Threshold probability", y = "Net benefit", colour = NULL,
         fill = NULL, title = "C  Decision curves") +
    theme_pub + theme(legend.position = "top")

  thresholds <- read_csv(file.path(data_dir, "threshold_metrics.csv"),
                         show_col_types = FALSE) %>%
    filter(cohort == "eicu_anycad")
  p4 <- ggplot(thresholds, aes(x = metric, y = value, fill = source)) +
    geom_col(position = "dodge", colour = dark, linewidth = 0.3, width = 0.7) +
    scale_fill_manual(values = c("Raw" = pink_light, "Calibrated" = pink)) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(x = NULL, y = "Metric value", fill = NULL,
         title = "D  10% threshold metrics") +
    theme_pub + theme(legend.position = "top")
  (p1 | p2) / (p3 | p4)
})
save_plot(figure6, "figure6_transport_utility", 10.5, 8.2)

figure7 <- local({
  distribution <- read_csv(file.path(data_dir, "risk_distribution.csv"),
                           show_col_types = FALSE)
  plot_distribution <- function(cohort_name, panel) {
    data <- filter(distribution, cohort == cohort_name)
    colors <- if (cohort_name == "eICU expanded") {
      c("Raw" = pink_light, "Recalibrated" = pink)
    } else {
      c("Raw" = blue_light, "Recalibrated" = blue_dark)
    }
    threshold_color <- if (cohort_name == "eICU expanded") pink else blue_dark
    ggplot(data, aes(x = left, y = proportion, fill = source,
                     colour = source)) +
      geom_col(position = "identity", width = 1 / 30, alpha = 0.45,
               colour = NA) +
      geom_vline(xintercept = 0.10, colour = threshold_color,
                 linetype = "dashed", linewidth = 0.55) +
      scale_fill_manual(values = colors) +
      scale_x_continuous(limits = c(0, 1), labels = number_format(accuracy = 0.1)) +
      labs(x = "Predicted risk", y = "Patient proportion", fill = NULL,
           title = paste0(panel, "  ", cohort_name)) +
      theme_pub + theme(legend.position = "top")
  }
  p1 <- plot_distribution("MIMIC-III", "A")
  p2 <- plot_distribution("eICU expanded", "B")

  decisions <- read_csv(file.path(data_dir, "decision_matrix.csv"),
                        show_col_types = FALSE)
  p3 <- ggplot(decisions, aes(x = source, y = proportion, fill = outcome)) +
    geom_col(position = "fill", colour = dark, linewidth = 0.25, width = 0.7) +
    facet_wrap(~cohort) +
    scale_fill_manual(values = c(
      "True positive" = blue_dark,
      "False positive" = pink,
      "True negative" = blue_light,
      "False negative" = pink_light
    )) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    labs(x = NULL, y = "Patient proportion", fill = NULL,
         title = "C  10% threshold decision composition") +
    theme_pub + theme(legend.position = "bottom")

  thresholds <- read_csv(file.path(data_dir, "threshold_metrics.csv"),
                         show_col_types = FALSE)
  thresholds <- thresholds %>%
    mutate(
      cohort = recode(cohort,
                      mimic3 = "MIMIC-III",
                      eicu_anycad = "eICU expanded"),
      metric = factor(
        metric,
        levels = c("SENSITIVITY", "SPECIFICITY", "PPV", "NPV"),
        labels = c("Sensitivity", "Specificity", "PPV", "NPV")
      )
    )
  p4 <- ggplot(thresholds, aes(x = metric, y = value,
                               colour = cohort, shape = source)) +
    geom_point(size = 2.6, position = position_dodge(width = 0.35)) +
    facet_wrap(~cohort) +
    scale_colour_manual(values = cohort_colors) +
    scale_shape_manual(values = c("Raw" = 17, "Calibrated" = 16)) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(x = NULL, y = "Metric value", colour = NULL, shape = NULL,
         title = "D  Operating characteristics at 10%") +
    theme_pub + theme(legend.position = "top")
  (p1 | p2) / (p3 | p4)
})
save_plot(figure7, "figure7_risk_and_decisions", 10.5, 8.2)

figure8 <- local({
  endpoint <- read_csv(file.path(data_dir, "endpoint_composition.csv"),
                       show_col_types = FALSE) %>%
    filter(database %in% c("mimic4", "mimic3", "eicu_anycad")) %>%
    mutate(
      cohort = recode(database,
                      mimic4 = "MIMIC-IV",
                      mimic3 = "MIMIC-III",
                      eicu_anycad = "eICU expanded")
    )
  p1 <- ggplot(endpoint, aes(x = cohort, y = coded_macce_n, fill = cohort)) +
    geom_col(colour = dark, linewidth = 0.5, width = 0.62) +
    geom_point(aes(y = overlap_total_n, colour = cohort), size = 2.7) +
    geom_segment(aes(xend = cohort, y = overlap_total_n, yend = coded_macce_n),
                 colour = mid, linewidth = 0.4) +
    scale_y_log10(labels = comma) +
    scale_fill_manual(values = cohort_colors, guide = "none") +
    scale_colour_manual(values = cohort_colors, guide = "none") +
    labs(x = NULL, y = "Events (log scale)",
         title = "A  Coded and anchor-overlap events") +
    theme_pub

  interval <- read_csv(file.path(data_dir, "interval_sensitivity.csv"),
                       show_col_types = FALSE)
  trop <- interval %>%
    filter(grepl("Troponin", Assumption), Cohort == "mimic4") %>%
    mutate(value = as.numeric(Value),
           ensemble = as.numeric(`Temporal ensemble AUROC`))
  p2 <- ggplot(trop, aes(x = value, y = ensemble)) +
    geom_line(colour = blue_dark, linewidth = 0.8) +
    geom_point(aes(size = `Anchor-overlap`), colour = blue_dark) +
    scale_size_area(max_size = 5) +
    scale_x_continuous(breaks = sort(unique(trop$value))) +
    labs(x = "Troponin threshold multiplier",
         y = "Temporal-ensemble AUROC", size = "Overlap events",
         title = "B  Troponin-threshold sensitivity") +
    theme_pub

  eicu <- interval %>%
    filter(grepl("eICU documentation", Assumption),
           Cohort == "eicu_anycad") %>%
    mutate(
      lag = as.numeric(Value),
      events = as.numeric(`Anchor-overlap`),
      auroc = as.numeric(`Temporal ensemble AUROC`)
    )
  lag_long <- bind_rows(
    eicu %>%
      transmute(lag, metric = "Anchor-overlap events",
                value = events, label = sprintf("%.0f", events)),
    eicu %>%
      transmute(lag, metric = "Ensemble AUROC",
                value = auroc, label = sprintf("%.2f", auroc))
  ) %>%
    group_by(metric) %>%
    mutate(scaled = value / max(value, na.rm = TRUE)) %>%
    ungroup()
  p3 <- ggplot(lag_long, aes(x = factor(lag), y = metric, fill = scaled)) +
    geom_tile(colour = "white", linewidth = 0.6) +
    geom_text(aes(label = label), family = "serif",
              colour = ink, size = 3.0) +
    scale_fill_gradient(low = pink_light, high = pink) +
    labs(x = "eICU documentation lag, hours", y = NULL, fill = NULL,
         title = "C  Documentation-lag invariance") +
    theme_pub + theme(legend.position = "none")

  rule_ranges <- interval %>%
    mutate(auroc = as.numeric(`Temporal ensemble AUROC`)) %>%
    filter(Cohort %in% c("mimic4", "mimic3", "eicu_anycad")) %>%
    mutate(cohort = recode(Cohort,
                           mimic4 = "MIMIC-IV",
                           mimic3 = "MIMIC-III",
                           eicu_anycad = "eICU expanded")) %>%
    group_by(cohort) %>%
    summarise(low = min(auroc, na.rm = TRUE),
              high = max(auroc, na.rm = TRUE),
              middle = mean(range(auroc, na.rm = TRUE)),
              .groups = "drop")
  p4 <- ggplot(rule_ranges, aes(y = cohort, x = middle)) +
    geom_segment(aes(x = low, xend = high, yend = cohort, colour = cohort),
                 linewidth = 1.0) +
    geom_point(aes(colour = cohort), size = 2.3) +
    geom_text(aes(label = sprintf("%.2f-%.2f", low, high)),
              hjust = -0.12, family = "serif", colour = ink, size = 2.9) +
    coord_cartesian(xlim = c(0.62, 0.82)) +
    scale_colour_manual(values = cohort_colors, guide = "none") +
    labs(x = "Temporal-ensemble AUROC range", y = NULL,
         title = "D  Rule-variant AUROC range") +
    theme_pub
  (p1 | p2) / (p3 | p4)
})
save_plot(figure8, "figure8_endpoint_ascertainment", 10.5, 8.2)

figure5 <- local({
  selection <- read_csv(file.path(data_dir, "model_selection.csv"),
                        show_col_types = FALSE) %>%
    mutate(plot_color = if_else(
      grepl("Joint nested", model), pink, blue_dark
    ))
  p1 <- ggplot(selection, aes(x = auroc, y = reorder(model, auroc))) +
    geom_point(aes(colour = plot_color), size = 2.7) +
    scale_colour_identity() +
    geom_segment(aes(x = 0.75, xend = auroc, yend = model),
                 colour = "#C9C9C9", linewidth = 0.7) +
    geom_text(aes(label = sprintf("%.2f", auroc)), hjust = -0.2,
              family = "serif", colour = ink, size = 2.9) +
    scale_x_continuous(limits = c(0.75, 0.77)) +
    labs(x = "AUROC", y = NULL, title = "A  Nested model selection") +
    theme_pub

  organ <- read_csv(file.path(data_dir, "organ_graph_controls.csv"),
                    show_col_types = FALSE) %>%
    mutate(plot_color = if_else(
      grepl("Random|Reversed|Permuted|No continuous", model),
      pink, blue_dark
    ))
  p2 <- ggplot(organ, aes(x = auroc, y = reorder(model, auroc))) +
    geom_point(aes(colour = plot_color), size = 2.5) +
    scale_colour_identity() +
    geom_segment(aes(x = 0.73, xend = auroc, yend = model),
                 colour = "#C9C9C9", linewidth = 0.7) +
    scale_x_continuous(limits = c(0.73, 0.77)) +
    labs(x = "AUROC", y = NULL, title = "B  Organ graph controls") +
    theme_pub

  contrasts <- data.frame(
    contrast = c("Nested ensemble vs XGB+TS",
                 "Joint nested vs XGB+TS",
                 "CT-OGT vs XGB+TS",
                 "Enhanced fusion vs XGB+TS"),
    estimate = c(0.0098, 0.0073, 0.0025, 0.0125),
    low = c(0.0071, 0.0043, -0.0030, 0.0073),
    high = c(0.0124, 0.0106, 0.0070, 0.0172)
  )
  p3 <- ggplot(contrasts, aes(x = estimate, y = reorder(contrast, estimate))) +
    geom_vline(xintercept = 0, colour = mid, linewidth = 0.4) +
    geom_vline(xintercept = 0.01, colour = pink,
               linetype = "dashed", linewidth = 0.5) +
    geom_errorbarh(aes(xmin = low, xmax = high), height = 0.18,
                   colour = dark, linewidth = 0.7) +
    geom_point(colour = dark, size = 2.4) +
    labs(x = "AUROC difference", y = NULL, title = "C  Paired contrasts") +
    theme_pub

  weights <- data.frame(
    model = c("Static XGBoost", "XGB + TS", "Transformer",
              "Dynamic graph", "Seven-organ token", "Temporal ensemble"),
    auroc = c(0.6414, 0.7537, 0.7593, 0.7562, 0.7504, 0.7640)
  ) %>%
    mutate(
      benchmark = 0.7537,
      delta = auroc - benchmark,
      direction = if_else(delta >= 0, "Above XGB+TS", "Below XGB+TS")
    )
  p4 <- ggplot(weights, aes(x = delta, y = reorder(model, delta),
                            colour = direction)) +
    geom_vline(xintercept = 0, colour = mid, linewidth = 0.45) +
    geom_segment(aes(x = 0, xend = delta, yend = model),
                 linewidth = 0.75) +
    geom_point(size = 2.6) +
    geom_text(aes(label = sprintf("%+.2f", delta)), hjust = -0.22,
              family = "serif", colour = ink, size = 2.8) +
    scale_colour_manual(values = c(
      "Above XGB+TS" = blue_dark,
      "Below XGB+TS" = pink
    )) +
    coord_cartesian(xlim = c(-0.13, 0.04)) +
    labs(x = "AUROC difference from XGB+TS", y = NULL, colour = NULL,
         title = "D  Incremental value versus XGB+TS") +
    theme_pub + theme(legend.position = "top")
  (p1 | p2) / (p3 | p4)
})
save_plot(figure5, "figure5_model_selection", 10.5, 7.8)

figure9 <- local({
  organ <- read_csv(file.path(data_dir, "organ_attention.csv"),
                    show_col_types = FALSE) %>%
    pivot_longer(cols = c(MACCE, `Event-free`), names_to = "group",
                 values_to = "attention")
  p1 <- ggplot(organ, aes(x = group, y = reorder(channel, attention),
                          fill = attention, colour = group)) +
    geom_tile(linewidth = 0.9) +
    scale_fill_gradient(low = "#F2F2F2", high = accent) +
    scale_colour_manual(values = c(
      "MACCE" = blue_dark,
      "Event-free" = pink
    ), guide = "none") +
    labs(x = NULL, y = NULL, fill = "Attention",
         title = "A  Organ-channel attention") +
    theme_pub +
    theme(axis.text.x = element_text(angle = 20, hjust = 1))

  temporal <- read_csv(file.path(data_dir, "temporal_attention.csv"),
                       show_col_types = FALSE) %>%
    pivot_longer(cols = c(MACCE, `Event-free`), names_to = "group",
                 values_to = "attention")
  p2 <- ggplot(temporal, aes(x = hour, y = attention, colour = group)) +
    geom_line(linewidth = 0.9) +
    scale_colour_manual(values = c("Event-free" = pink, "MACCE" = blue_dark)) +
    labs(x = "ICU hour", y = "Mean temporal attention", colour = NULL,
         title = "B  Hour-specific attention") +
    theme_pub + theme(legend.position = "top")

  contrast <- organ %>%
    group_by(channel, group) %>%
    summarise(attention = mean(attention), .groups = "drop") %>%
    pivot_wider(names_from = group, values_from = attention) %>%
    mutate(difference = MACCE - `Event-free`) %>%
    slice_max(abs(difference), n = 10)
  p3 <- ggplot(contrast, aes(x = difference, y = reorder(channel, difference))) +
    geom_vline(xintercept = 0, colour = mid, linewidth = 0.4) +
    geom_col(aes(fill = difference > 0), colour = dark, linewidth = 0.3,
             width = 0.65) +
    scale_fill_manual(values = c("TRUE" = blue_dark, "FALSE" = pink)) +
    labs(x = "MACCE minus event-free attention", y = NULL, fill = NULL,
         title = "C  Top attention contrasts") +
    theme_pub + theme(legend.position = "none")

  (p1 | p2) / p3 + plot_layout(heights = c(1.2, 0.8))
})
save_plot(figure9, "figure9_attention", 10.5, 8.2)

cat("R figures written to", out_dir, "\n")
