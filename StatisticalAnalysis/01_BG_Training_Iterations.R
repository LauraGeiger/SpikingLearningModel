# Install packages if not already installed
install.packages("tidyverse")
install.packages("readxl")
install.packages("ggplot2")
remove.packages("rlang")
install.packages("rlang")
install.packages("ggstatsplot")


# Load packages
library(tidyverse)
library(readxl)
library(ggplot2)
library(ggstatsplot)
library(patchwork)

data <- read_excel("01_Exp_BG_Training_Iterations.xlsx")

data <- data %>%
  mutate(
    Iterations = as.numeric(Iterations),
    StopIteration = as.numeric(StopIteration),
    Loop = as.factor(Loop),
    GroupNameLabel = gsub(" ", "\n", GroupName),
    GroupFlexion = ifelse(
      is.na(`Flexion threshold`),
      GroupNameLabel,
      paste0(GroupNameLabel, "\n(th = ", `Flexion threshold`, ")")
    )#,
    #GroupNameOrdered = factor(GroupFlexion, levels = unique(GroupFlexion[order(Group, `Flexion threshold`)]))
  )

# create factor levels ordered by numeric Group
data <- data %>%
  arrange(Group) %>%
  mutate(
    # preserve thresholds in labels, but order by Group number
    GroupNameOrdered = factor(GroupFlexion, levels = unique(GroupFlexion[order(Group)]))
  )

# --- Motor loop plot ---
data_motor <- data %>% filter(Loop == "Motor")

ggstatsplot::ggbetweenstats(
  data = data_motor,
  x = GroupNameOrdered,             # experiment type
  y = Iterations,
  type = "parametric",   # or "nonparametric"
  results.subtitle = FALSE,
  pairwise.display = "none",
  plot.type = "violin",
  ggtheme = ggplot2::theme_minimal(),
  title = "Motor Loop: Iterations required for learning"
) + labs(x = NULL)

# --- Premotor loop plot ---
data_premotor <- data %>% filter(Loop == "Premotor")

ggstatsplot::ggbetweenstats(
  data = data_premotor,
  x = GroupNameOrdered,
  y = Iterations,
  type = "parametric",
  results.subtitle = FALSE,
  pairwise.display = "none",
  plot.type = "violin",
  ggtheme = ggplot2::theme_minimal(),
  title = "Premotor Loop: Iterations required for learning"
) + labs(x = NULL) 

ggstatsplot::ggbetweenstats(
  data = data_premotor,
  x = GroupNameOrdered,
  y = StopIteration,
  type = "parametric",
  results.subtitle = FALSE,
  pairwise.display = "none",
  plot.type = "violin",
  ggtheme = ggplot2::theme_minimal(),
  title = "Both Loops: Iterations required for learning"
) + labs(x = NULL, y = "Iterations") 


# Function to make plots per reward criteria
make_plot <- function(df, loop_type, y_var, reward) {
  ggstatsplot::ggbetweenstats(
    data = df %>% filter(Loop == loop_type, 'Reward criteria' == reward),
    x = GroupNameOrdered,
    y = !!sym(y_var),
    type = "parametric",
    results.subtitle = FALSE,
    pairwise.display = "none",
    plot.type = "violin",
    ggtheme = ggplot2::theme_minimal(),
    title = paste(loop_type, "Loop:", y_var, "\nReward Criteria = ", reward)
  ) + labs(x = NULL, y = y_var)
}

# Get list of reward criteria
reward_list <- unique(data$`Reward criteria`)

# --- Compute ranges ---
motor_range <- range(data %>% filter(Loop == "Motor") %>% pull(Iterations), na.rm = TRUE)
premotor_range <- range(data %>% filter(Loop == "Premotor") %>% pull(Iterations), na.rm = TRUE)

# --- Helper function ---
make_plot <- function(df, loop_type, reward, y_limits) {
  if (length(unique(df$GroupFlexion)) >= 2) {
    ggstatsplot::ggbetweenstats(
      data = df,
      x = GroupFlexion,
      y = Iterations,
      type = "parametric",
      results.subtitle = FALSE,
      pairwise.display = "none",
      plot.type = "violin",
      ggtheme = ggplot2::theme_minimal(),
      title = paste(loop_type, "Loop\nReward =", reward)
    ) + labs(x = NULL, y = "Iterations") +
      ylim(y_limits)  
  } else {
    NULL
  }
}

# --- Motor plots ---
plots_motor <- lapply(reward_list, function(reward) {
  df <- data %>% filter(Loop == "Motor", `Reward criteria` == reward)
  make_plot(df, "Motor", reward, motor_range)
})

# --- Premotor plots ---
plots_premotor <- lapply(reward_list, function(reward) {
  df <- data %>% filter(Loop == "Premotor", `Reward criteria` == reward)
  make_plot(df, "Premotor", reward, premotor_range)
})

# Remove NULLs
plots_motor <- Filter(Negate(is.null), plots_motor)
plots_premotor <- Filter(Negate(is.null), plots_premotor)

# --- Arrange rows ---
final_plot <- (wrap_plots(plots_motor, nrow = 1)) /
  (wrap_plots(plots_premotor, nrow = 1))

final_plot
