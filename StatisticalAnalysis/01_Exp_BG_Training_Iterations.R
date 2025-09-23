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

data <- read_excel("01_Exp_BG_Training_Iterations.xlsx")

data <- data %>%
  mutate(
    Iterations = as.numeric(Iterations),
    Loop = as.factor(Loop),
    GroupNameLabel = gsub(" ", "\n", GroupName),
    GroupNameOrdered = factor(GroupNameLabel, levels = unique(GroupNameLabel[order(Group)]))
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


