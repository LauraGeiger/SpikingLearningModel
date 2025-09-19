# Install packages if not already installed
install.packages("tidyverse")
install.packages("readxl")
install.packages("ggplot2")

# Load packages
library(tidyverse)
library(readxl)
library(ggplot2)

#windows()

data <- read_excel("01_Exp_BG_Training_Iterations.xlsx")

# Inspect the data
glimpse(data)

data <- data %>%
  mutate(Iterations = as.numeric(Iterations))

ggplot(data %>% filter(!is.na(Iterations)), 
        aes(x = Loop, y = Iterations, fill = Group)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  facet_wrap(~Group) +
  labs(title = "Iterations required for learning per loop type",
       x = "Loop Type",
       y = "Iterations") +
  theme_minimal()




summary_data <- data %>%
  group_by(Group, Loop) %>%
  summarize(avg_iterations = mean(Iterations, na.rm = TRUE))

print(ggplot(summary_data, aes(x = Loop, y = avg_iterations, fill = Group)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "Average iterations required per loop type",
       x = "Loop Type",
       y = "Average Iterations") +
  theme_minimal())


