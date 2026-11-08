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
accent <- "#B05A4A"

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
    panel.grid.major = element_line(colour = "#D9D9D9", linewidth = 0.35),
    panel.grid.minor = element_blank(),
    plot.background = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA)
  )

save_plot <- function(plot, stem, width = 8.5, height = 5.5) {
  ggsave(
    file.path(out_dir, paste0(stem, ".png")),
    plot,
    width = width,
    height = height,
    dpi = 320,
    bg = "white"
  )
  ggsave(
    file.path(out_dir, paste0(stem, ".pdf")),
    plot,
    width = width,
    height = height,
    bg = "white"
  )
}

figure1 <- local({
  nodes <- data.frame(
    id = 1:10,
    x = c(1.5, 4.25, 7.0, 9.75, 5.6, 3.0, 8.2, 5.6, 5.6, 5.6),
    y = c(8.0, 8.0, 8.0, 8.0, 6.4, 4.8, 4.8, 3.2, 1.8, 0.4),
    label = c(
      "MIMIC-IV\n2008-2019",
      "MIMIC-III\ntransport",
      "eICU expanded\nCAD cohort",
      "24-hour\nlandmark",
      "Static variables\nand organ trajectories",
      "XGB + temporal\nsummaries",
      "Transformer and\norgan graphs",
      "2017-2019 internal\nmodel selection",
      "Locked 2020-2022\ntemporal test",
      "Calibration, utility,\ntransport and endpoint\nsensitivity"
    )
  )
  edges <- data.frame(
    x = c(2.4, 5.1, 7.8, 10.0, 6.8, 4.2, 8.8, 5.6, 5.6),
    y = c(7.7, 7.7, 7.7, 7.6, 6.0, 5.4, 5.4, 4.4, 2.8),
    xend = c(6.7, 8.3, 9.5, 5.9, 6.0, 5.4, 7.4, 5.6, 5.6),
    yend = c(6.6, 6.6, 6.6, 6.8, 5.1, 4.9, 4.9, 3.5, 2.1)
  )
  ggplot() +
    geom_segment(
      data = edges,
      aes(x = x, y = y, xend = xend, yend = yend),
      arrow = arrow(length = unit(0.14, "inches")),
      linewidth = 0.55,
      colour = mid
    ) +
    geom_rect(
      data = nodes,
      aes(xmin = x - 1.05, xmax = x + 1.05, ymin = y - 0.45, ymax = y + 0.45),
      fill = light,
      colour = dark,
      linewidth = 0.55
    ) +
    geom_text(
      data = nodes,
      aes(x = x, y = y, label = label),
      family = "serif",
      colour = ink,
      size = 3.0,
      lineheight = 0.92
    ) +
    coord_cartesian(xlim = c(0, 11), ylim = c(-0.2, 8.7), clip = "off") +
    annotate("text", x = 0.05, y = 8.65, label = "A", family = "serif",
             fontface = "bold", colour = ink, size = 5) +
    theme_void(base_family = "serif") +
    theme(
      plot.background = element_rect(fill = "white", colour = NA),
      text = element_text(colour = ink)
    )
})
save_plot(figure1, "figure1_technical_route", 10, 6.2)

figure2 <- local({
  cohort <- data.frame(
    cohort = factor(
      c("MIMIC-IV", "MIMIC-III", "eICU expanded"),
      levels = c("MIMIC-IV", "MIMIC-III", "eICU expanded")
    ),
    n = c(13131, 3470, 5913),
    events = c(4475, 409, 642)
  )
  p1 <- ggplot(cohort, aes(x = cohort, y = n)) +
    geom_col(fill = light, colour = dark, linewidth = 0.55, width = 0.58) +
    geom_point(aes(y = events), colour = accent, size = 3.2) +
    geom_segment(aes(xend = cohort, y = events, yend = n),
                 colour = mid, linewidth = 0.45) +
    scale_y_log10(labels = comma) +
    labs(x = NULL, y = "Patients (log scale)", title = "A  Cohort size and events") +
    theme_pub
  components <- data.frame(
    cohort = rep(c("MIMIC-IV", "MIMIC-III", "eICU expanded"), each = 3),
    component = rep(c("Acute MI", "Stroke", "Death"), times = 3),
    events = c(3214, 1079, 524, 219, 199, 13, 459, 119, 90)
  )
  p2 <- ggplot(components, aes(x = cohort, y = events, fill = component)) +
    geom_col(position = "dodge", colour = dark, linewidth = 0.35, width = 0.72) +
    scale_fill_manual(values = c("#FFFFFF", "#BDBDBD", accent)) +
    labs(x = NULL, y = "Recorded events", fill = NULL,
         title = "B  Recorded outcome components") +
    theme_pub +
    theme(legend.position = "top")
  p1 / p2 + plot_layout(heights = c(1, 1))
})
save_plot(figure2, "figure2_study_design", 8.6, 7.4)

figure3 <- local({
  nodes <- data.frame(
    x = c(1.2, 3.3, 5.5, 7.8, 10.0),
    y = c(3.5, 3.5, 3.5, 3.5, 3.5),
    label = c(
      "7 organ systems\n20 channels",
      "Observation masks\nelapsed time",
      "Organ or channel\ndynamic graph",
      "Transformer\nencoder",
      "7-day MACCE\nrisk"
    )
  )
  ggplot() +
    geom_segment(
      aes(x = c(2.1, 4.2, 6.55, 8.9), y = 3.5,
          xend = c(2.65, 4.75, 7.1, 9.35), yend = 3.5),
      arrow = arrow(length = unit(0.13, "inches")),
      colour = mid,
      linewidth = 0.55
    ) +
    geom_rect(
      data = nodes,
      aes(xmin = x - 0.85, xmax = x + 0.85, ymin = 2.85, ymax = 4.15),
      fill = light, colour = dark, linewidth = 0.55
    ) +
    geom_text(
      data = nodes,
      aes(x = x, y = y, label = label),
      family = "serif", colour = ink, size = 3.1, lineheight = 0.95
    ) +
    coord_cartesian(xlim = c(0, 11), ylim = c(2.3, 4.8), clip = "off") +
    annotate("text", x = 0.1, y = 4.7, label = "A", family = "serif",
             fontface = "bold", colour = ink, size = 5) +
    theme_void(base_family = "serif") +
    theme(plot.background = element_rect(fill = "white", colour = NA))
})
save_plot(figure3, "figure3_model_architecture", 10, 4.1)

figure4 <- local({
  dev <- read_csv(file.path(data_dir, "development_performance.csv"),
                  show_col_types = FALSE) %>%
    filter(model %in% c("xgb_static", "xgb_static_ts", "transformer",
                        "ct_graph", "organ_graph", "temporal_ensemble")) %>%
    mutate(
      label = recode(
        model,
        xgb_static = "Static XGBoost",
        xgb_static_ts = "XGB + temporal summaries",
        transformer = "Transformer",
        ct_graph = "Dynamic organ graph",
        organ_graph = "Seven-organ token graph",
        temporal_ensemble = "Temporal ensemble"
      )
    )
  p1 <- ggplot(dev, aes(x = auroc, y = reorder(label, auroc))) +
    geom_segment(aes(x = 0.60, xend = auroc, yend = label),
                 colour = "#C9C9C9", linewidth = 0.8) +
    geom_point(colour = accent, size = 2.8) +
    geom_text(aes(label = sprintf("%.2f", auroc)), hjust = -0.28,
              family = "serif", colour = ink, size = 2.9) +
    scale_x_continuous(limits = c(0.60, 0.80)) +
    labs(x = "AUROC", y = NULL, title = "A  Development discrimination") +
    theme_pub
  strong <- read_csv(file.path(data_dir, "strong_holdout.csv"),
                     show_col_types = FALSE)
  p2 <- ggplot(strong, aes(x = reorder(model, auroc), y = auroc)) +
    geom_col(fill = light, colour = dark, linewidth = 0.5, width = 0.62) +
    geom_text(aes(label = sprintf("%.2f", auroc)), vjust = -0.3,
              family = "serif", colour = ink, size = 3.0) +
    coord_cartesian(ylim = c(0.68, 0.77)) +
    labs(x = NULL, y = "AUROC", title = "B  Locked 2020-2022 test") +
    theme_pub +
    theme(axis.text.x = element_text(angle = 20, hjust = 1))
  p1 | p2
})
save_plot(figure4, "figure4_performance", 10, 5.6)

figure5 <- local({
  calibration <- read_csv(file.path(data_dir, "calibration_curves.csv"),
                          show_col_types = FALSE) %>%
    mutate(cohort = recode(cohort, mimic3 = "MIMIC-III",
                           eicu_anycad = "eICU expanded CAD"))
  reliability <- function(cohort_name, panel) {
    ggplot(filter(calibration, cohort == cohort_name),
           aes(x = predicted, y = observed, colour = source,
               linetype = source)) +
      geom_abline(slope = 1, intercept = 0, colour = mid, linewidth = 0.45) +
      geom_line(linewidth = 0.8) +
      geom_point(size = 2.2) +
      scale_colour_manual(values = c("Raw" = mid, "Recalibrated" = accent)) +
      scale_x_continuous(limits = c(0, 1), labels = number_format(accuracy = 0.1)) +
      scale_y_continuous(limits = c(0, 1), labels = number_format(accuracy = 0.1)) +
      labs(x = "Predicted risk", y = "Observed risk", colour = NULL,
           linetype = NULL, title = paste0(panel, "  ", cohort_name)) +
      theme_pub + theme(legend.position = "top")
  }
  p1 <- reliability("MIMIC-III", "A")
  p2 <- reliability("eICU expanded CAD", "B")
  dca <- read_csv(file.path(data_dir, "decision_curves.csv"),
                  show_col_types = FALSE) %>%
    filter(cohort == "eicu_anycad")
  p3 <- ggplot(dca, aes(x = threshold, y = net_benefit, colour = source,
                        fill = source)) +
    geom_ribbon(aes(ymin = low, ymax = high), alpha = 0.13,
                colour = NA) +
    geom_line(linewidth = 0.9) +
    geom_hline(yintercept = 0, linewidth = 0.4, colour = ink) +
    scale_colour_manual(values = c("Raw" = mid, "Recalibrated" = accent)) +
    scale_fill_manual(values = c("Raw" = mid, "Recalibrated" = accent)) +
    labs(x = "Threshold probability", y = "Net benefit", colour = NULL,
         fill = NULL, title = "C  Decision curves in expanded eICU") +
    theme_pub + theme(legend.position = "top")
  thresholds <- read_csv(file.path(data_dir, "threshold_metrics.csv"),
                         show_col_types = FALSE) %>%
    filter(cohort == "eicu_anycad")
  p4 <- ggplot(thresholds, aes(x = metric, y = value, fill = source)) +
    geom_col(position = "dodge", colour = dark, linewidth = 0.35, width = 0.7) +
    scale_fill_manual(values = c("Raw" = "#BDBDBD", "Calibrated" = accent)) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(x = NULL, y = "Metric value", fill = NULL,
         title = "D  10% threshold metrics in expanded eICU") +
    theme_pub + theme(legend.position = "top")
  (p1 | p2) / (p3 | p4)
})
save_plot(figure5, "figure5_calibration_utility", 10, 8.2)

figure6 <- local({
  interval <- read_csv(file.path(data_dir, "interval_sensitivity.csv"),
                       show_col_types = FALSE)
  eicu <- interval %>%
    filter(grepl("eICU documentation", Assumption),
           Cohort == "eicu_anycad") %>%
    mutate(value = as.numeric(Value))
  p1 <- ggplot(eicu, aes(x = value, y = `Anchor-overlap`)) +
    geom_area(fill = light, colour = NA, alpha = 0.9) +
    geom_line(colour = dark, linewidth = 0.8) +
    geom_point(colour = accent, size = 2.5) +
    scale_x_continuous(breaks = sort(unique(eicu$value))) +
    labs(x = "eICU documentation lag, hours", y = "Anchor-overlap events",
         title = "A  Documentation-lag sensitivity") +
    theme_pub
  trop <- interval %>%
    filter(grepl("Troponin", Assumption), Cohort == "mimic4") %>%
    mutate(value = as.numeric(Value),
           ensemble = as.numeric(`Temporal ensemble AUROC`))
  p2 <- ggplot(trop, aes(x = value, y = ensemble)) +
    geom_line(colour = dark, linewidth = 0.8) +
    geom_point(aes(size = `Anchor-overlap`), colour = accent) +
    scale_size_area(max_size = 5) +
    scale_x_continuous(breaks = sort(unique(trop$value))) +
    labs(x = "Troponin threshold multiplier", y = "Temporal-ensemble AUROC",
         size = "Overlap events", title = "B  Troponin-threshold sensitivity") +
    theme_pub + theme(legend.position = "right")
  p1 | p2
})
save_plot(figure6, "figure6_interval_sensitivity", 10, 4.8)

figure7 <- local({
  organ <- read_csv(file.path(data_dir, "organ_attention.csv"),
                    show_col_types = FALSE) %>%
    pivot_longer(cols = c(MACCE, `Event-free`), names_to = "group",
                 values_to = "attention")
  p1 <- ggplot(organ, aes(x = group, y = reorder(channel, attention),
                          fill = attention)) +
    geom_tile(colour = "white", linewidth = 0.45) +
    scale_fill_gradient(low = "#F2F2F2", high = accent) +
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
    scale_colour_manual(values = c("Event-free" = mid, "MACCE" = accent)) +
    labs(x = "ICU hour", y = "Mean temporal attention", colour = NULL,
         title = "B  Hour-specific attention") +
    theme_pub + theme(legend.position = "top")
  p1 | p2 + plot_layout(widths = c(1.15, 1))
})
save_plot(figure7, "figure7_attention", 10, 6.1)

cat("R figures written to", out_dir, "\n")
