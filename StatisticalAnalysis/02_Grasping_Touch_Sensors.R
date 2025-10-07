library(readxl)
library(tidyr)
library(dplyr)
library(ggplot2)
library(purrr)

# Base path 
path_without_object <- "../Data/Experiments/02_Grasping/007/"
path_with_object <- "../Data/Experiments/02_Grasping/014/"

# Find files
file_without_object <- list.files(path_without_object, pattern = "MotorLearning\\.xlsx$", full.names = TRUE)[1]
file_with_object <- list.files(path_with_object, pattern = "MotorLearning\\.xlsx$", full.names = TRUE)[1]

# Load grasping data if files exist
grasping_without_object <- if (!is.na(file_without_object)) {
  read_excel(file_without_object, sheet="sensor_touch_grasp_over_time") %>%
    pivot_longer(
      cols = c(`0`, `1`, `2`),     # the sensor columns
      names_to = "sensor", 
      values_to = "value"
    ) %>%
    mutate(iteration = time / 100)
} else {
  NULL
}
grasping_with_object <- if (!is.na(file_with_object)) {
  read_excel(file_with_object, sheet="sensor_touch_grasp_over_time") %>%
    pivot_longer(
      cols = c(`0`, `1`, `2`),     # the sensor columns
      names_to = "sensor", 
      values_to = "value"
    ) %>%
    mutate(iteration = time / 100)
} else {
  NULL
}

# Load holding data if files exist
holding_without_object <- if (!is.na(file_without_object)) {
  read_excel(file_without_object, sheet="sensor_touch_hold_over_time") %>%
    pivot_longer(
      cols = c(`0`, `1`, `2`),     # the sensor columns
      names_to = "sensor", 
      values_to = "value"
    ) %>%
    mutate(iteration = time / 100)
} else {
  NULL
}
holding_with_object <- if (!is.na(file_with_object)) {
  read_excel(file_with_object, sheet="sensor_touch_hold_over_time") %>%
    pivot_longer(
      cols = c(`0`, `1`, `2`),     # the sensor columns
      names_to = "sensor", 
      values_to = "value"
    ) %>%
    mutate(iteration = time / 100)
} else {
  NULL
}

# Load feedback data if files exist
feedback_without_object <- if (!is.na(file_without_object)) {
  read_excel(file_without_object, sheet="feedback_over_time") %>%
    mutate(iteration = time / 100)
} else {
  NULL
} 
feedback_with_object <- if (!is.na(file_with_object)) {
  read_excel(file_with_object, sheet="feedback_over_time") %>%
    mutate(iteration = time / 100)
} else {
  NULL
} 

# Merge grasping data with feedback
grasping_with_object_fb <- grasping_with_object %>%
  left_join(feedback_with_object %>% select(iteration, grasp), by = "iteration")

grasping_without_object_fb <- grasping_without_object %>%
  left_join(feedback_without_object %>% select(iteration, grasp), by = "iteration")

# Merge holding data with feedback
holding_with_object_fb <- holding_with_object %>%
  left_join(feedback_with_object %>% select(iteration, hold), by = "iteration")

holding_without_object_fb <- holding_without_object %>%
  left_join(feedback_without_object %>% select(iteration, hold), by = "iteration")



ggstatsplot::ggbetweenstats(
  data = grasping_without_object,
  x = sensor,             
  y = value,
  type = "parametric",   # or "nonparametric"
  results.subtitle = FALSE,
  pairwise.display = "none",
  plot.type = "violin",
  ggtheme = ggplot2::theme_minimal(),
  title = "Touch sensor values from grasping without object"
) + labs(x = "Touch Sensor", y = "Value difference before and after grasping")

ggstatsplot::ggbetweenstats(
  data = grasping_with_object,
  x = sensor,             
  y = value,
  type = "parametric",   # or "nonparametric"
  results.subtitle = FALSE,
  pairwise.display = "none",
  plot.type = "violin",
  ggtheme = ggplot2::theme_minimal(),
  title = "Touch sensor values from grasping with object"
) + labs(x = "Touch Sensor", y = "Value difference before and after grasping")

grasping_with_object_fb %>%
  mutate(sensor_feedback = paste0("Sensor ", sensor, "\nGrasp=", grasp)) %>%
  ggstatsplot::ggbetweenstats(
    data = .,
    x = sensor_feedback,
    y = value,
    type = "parametric",
    results.subtitle = FALSE,
    pairwise.display = "none",
    plot.type = "violin",
    ggtheme = ggplot2::theme_minimal(),
    title = "Touch sensor values (grasping with object)"
  ) + labs(x = "Sensor × Grasp", y = "Sensor Value")


ggstatsplot::ggbetweenstats(
  data = holding_without_object,
  x = sensor,             
  y = value,
  type = "parametric",   # or "nonparametric"
  results.subtitle = FALSE,
  pairwise.display = "none",
  plot.type = "violin",
  ggtheme = ggplot2::theme_minimal(),
  title = "Touch sensor values from grasping without object"
) + labs(x = "Touch Sensor", y = "Value difference before and after holding")

ggstatsplot::ggbetweenstats(
  data = holding_with_object,
  x = sensor,             
  y = value,
  type = "parametric",   # or "nonparametric"
  results.subtitle = FALSE,
  pairwise.display = "none",
  plot.type = "violin",
  ggtheme = ggplot2::theme_minimal(),
  title = "Touch sensor values from grasping with object"
) + labs(x = "Touch Sensor", y = "Value difference before and after holding")

holding_with_object_fb %>%
  mutate(sensor_feedback = paste0("Sensor ", sensor, "\nHold=", hold)) %>%
  ggstatsplot::ggbetweenstats(
    data = .,
    x = sensor_feedback,
    y = value,
    type = "parametric",
    results.subtitle = FALSE,
    pairwise.display = "none",
    plot.type = "violin",
    ggtheme = ggplot2::theme_minimal(),
    title = "Touch sensor values (holding with object)"
  ) + labs(x = "Sensor × Hold", y = "Sensor Value")
