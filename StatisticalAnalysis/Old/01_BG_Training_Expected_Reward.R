library(readxl)
library(tidyr)
library(dplyr)
library(ggplot2)
library(purrr)

# Base path containing experiment folders
base_path <- "../Data/Experiments/01_BG_Training/1_RewardCriteria_ExpectedReward/01_SIM/"

# List all experiment subfolders (e.g., "001", "002", ...)
exp_folders <- list.dirs(base_path, full.names = TRUE, recursive = FALSE)

# Helper function to load both MotorLoop and PremotorLoop from one experiment
load_experiment <- function(folder) {
  exp_id <- basename(folder)  # e.g., "001"
  
  # Find files
  motor_file <- list.files(folder, pattern = "MotorLoop\\.xlsx$", full.names = TRUE)[1]
  premotor_file <- list.files(folder, pattern = "PremotorLoop\\.xlsx$", full.names = TRUE)[1]
  
  # Load data if files exist
  motor <- if (!is.na(motor_file)) {
    read_excel(motor_file, sheet="expected_reward_over_time") %>%
      pivot_longer(-time, names_to = "action", values_to = "reward") %>%
      filter(action != "0000") %>%
      mutate(loop = "Motor Loop", experiment = exp_id)
  } else {
    NULL
  }
  
  premotor <- if (!is.na(premotor_file)) {
    read_excel(premotor_file, sheet="expected_reward_over_time") %>%
      pivot_longer(-time, names_to = "action", values_to = "reward") %>%
      filter(action != "000") %>%
      mutate(loop = "Premotor Loop", experiment = exp_id)
  } else {
    NULL
  }
  
  bind_rows(motor, premotor)
}

# Apply function to all experiment folders
all_data <- map_dfr(exp_folders, load_experiment)

all_data <- all_data %>%
  mutate(iteration = time / 100, 
    reward = reward * 100)


ggplot(all_data, aes(
    x = iteration, 
    y = reward, 
    color = loop,
    group = interaction(loop, action)
  )) +
  geom_line(size = 1) +
  facet_wrap(~experiment) + 
  theme_minimal(base_size = 14) +
  labs(
    title = "Expected Reward over Iterations",
    x = "Iteration",
    y = "Expected Reward (%)",
    color = NULL
  )


