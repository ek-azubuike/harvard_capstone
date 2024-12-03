# environment setup, loading required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(Hmisc)) install.packages("Hmisc", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dlabs", repos = "http://cran.us.r-project.org")
if(!require(GGally)) install.packages("GGally", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(moments)) install.packages("moments", repos = "http://cran.us.r-project.org")

ds_theme_set()

options(timeout = 120)

# download and load data set
dl <- "wine_quality.zip"
if(!file.exists(dl))
  download.file("https://archive.ics.uci.edu/static/public/186/wine+quality.zip", dl)

red_file <- "winequality-red.csv"
if(!file.exists(red_file))
  unzip(dl, red_file)

white_file <- "winequality-white.csv"
if(!file.exists(white_file))
  unzip(dl, white_file)

red.df <- read_delim(file = red_file,
                     delim = ";",
                     col_types = list(
                       quality = col_integer(),
                       .default = col_double()))
red.df$color <- "red"

white.df <- read_delim(file = white_file,
                       delim = ";",
                       col_types = list(
                         quality = col_integer(),
                         .default = col_double()))
white.df$color = "white"

names(red.df) <- make.names(colnames(red.df))
names(white.df) <- make.names(colnames(white.df))

wine.df <- rbind(red.df, white.df)
wine.df <- wine.df %>%
  mutate(color = as_factor(color))

wine.x <- wine.df %>% 
  select(-quality)
wine.y <- wine.df %>% 
  select(quality)

# data pre-processing
set.seed(2024)

# create train and test sets
test.index <- createDataPartition(wine.y$quality,
                                  p = 0.2,
                                  list = FALSE)
wine.x.train <- wine.x[-test.index,]
wine.x.test <- wine.x[test.index,]
wine.y.train <- wine.y[-test.index,]
wine.y.test <- wine.y[test.index,]

head(wine.x.train)
summary(wine.x.train)

# exploratory data analysis

# boxplots of variables
wine.x.train %>% 
  select(-color) %>% 
  mutate_all(scale) %>% 
  pivot_longer(cols = names(wine.x.train)[-12]) %>% 
  group_by(name) %>% 
  ggplot(aes(x = name, y = value)) +
  geom_boxplot() +
  coord_flip()

# skew of variables
wine.x.train %>% 
  select(-color) %>% 
  pivot_longer(cols = names(wine.x.train)[-12]) %>% 
  group_by(name) %>%
  summarise(skew = skewness(value)) %>% 
  filter(skew > 1 | skew < -1)

# log transformation of data to remove skewness
high.skew.ind <- c("chlorides", "fixed.acidity", "free.sulfur.dioxide", 
               "residual.sugar", "sulphates", "volatile.acidity")
low.skew.train <- wine.x.train[names(wine.x.train)[!names(wine.x.train) %in% high.skew.ind]]
low.skew.test <- wine.x.test[names(wine.x.test)[!names(wine.x.test) %in% high.skew.ind]]

wine.x.train %>% 
  select(all_of(high.skew.ind)) %>% 
  pivot_longer(cols = high.skew.ind) %>%
  mutate(value = ifelse(value == 0, 0, log(value))) %>%
  group_by(name) %>%
  summarise(skew = skewness(value))

# log transform data 
wine.x.train.scaled <- sapply(wine.x.train[high.skew.ind], function(x){
  ifelse(x == 0, 0, log(x)) }) %>% 
  cbind(low.skew.train) %>%
  mutate(color = factor(color)) %>% 
  data.frame()

wine.x.test.scaled <- sapply(wine.x.test[high.skew.ind], function(x){
  ifelse(x == 0, 0, log(x)) }) %>% 
  cbind(low.skew.test) %>% 
  mutate(color = factor(color)) %>% 
  data.frame()

# skew of log-transformed variables
wine.x.train.scaled %>% 
  select(-color) %>% 
  pivot_longer(cols = names(wine.x.train.scaled)[-12]) %>% 
  group_by(name) %>%
  summarise(skew = skewness(value))

# box plots of log-transformed variables
wine.x.train.scaled %>% 
  select(-color) %>% 
  mutate_all(scale) %>% 
  pivot_longer(cols = names(wine.x.train.scaled)[-12]) %>% 
  group_by(name) %>% 
  ggplot(aes(x = name, y = value)) +
  geom_boxplot() +
  coord_flip()

# pairwise correlations
ggpairs(wine.x.train.scaled,
        lower = list(continuous = wrap("points", alpha = 0.05)),
        mapping = aes(color = color))

# correlation plot
ggcorr(wine.x.train.scaled,
       size = 3,
       hjust = 0.7,
       label = TRUE)

# variable selection
wine.train <- tibble(cbind(wine.x.train.scaled, wine.y.train$quality)) %>% 
  rename(quality = `wine.y.train$quality`)
wine.train

wine.train.selected <- wine.train %>% 
  mutate(alcohol.density = alcohol * density) %>% 
  select(-c(alcohol, density, free.sulfur.dioxide))
wine.train.selected

wine.test <- tibble(cbind(wine.x.test.scaled, wine.y.test$quality)) %>% 
  rename(quality = `wine.y.test$quality`) 
wine.test

wine.test.selected <-  wine.test %>%
  mutate(alcohol.density = alcohol * density) %>% 
  select(-c(alcohol, density, free.sulfur.dioxide))
wine.test.selected

# modeling

# base ("naive") model
mu <- mean(as.numeric(wine.train$quality))

base.pred <- rep(mu, 
                 times = length(wine.test$quality))
base.rmse <- RMSE(base.pred, as.numeric(wine.test$quality))
base.rmse

# generalized linear model
wine.lm <- train(quality ~ .,
                 data = wine.train.selected,
                 preProcess = "center",
                 method = "glm")

lm.rmse <- RMSE(predict(wine.lm, wine.test.selected),
                wine.test.selected$quality)
lm.rmse

# knn
wine.train <- wine.train %>% 
  mutate(quality = factor(quality))

wine.test <- wine.test %>% 
  mutate(quality = factor(quality))

quality.levels <- levels(wine.test$quality)
controls <- trainControl(method = "cv", 
                         p = 0.8, 
                         number = 10)
grid = data.frame(k = seq(5, 25, 2))

wine.knn <- train(quality ~ .,
                  method = "knn",
                  data = wine.train,
                  trControl = controls,
                  tuneGrid = grid)

wine.knn$bestTune

knn.preds <- factor(predict(wine.knn, wine.test),
                    levels = levels(wine.test$quality))
knn.cm <- confusionMatrix(knn.preds, wine.test$quality)
knn.cm

# random tree
wine.rt <- train(quality ~ .,
                 data = wine.train,
                 tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
                 method = "rpart")

wine.rt$bestTune

rt.preds <- factor(predict(wine.rt, wine.test),
                   levels = levels(wine.test$quality))

rt.cm <- confusionMatrix(rt.preds, wine.test$quality)
rt.cm

# random forest
controls <- trainControl(method = "cv", p = 0.8, number = 5)

wine.rf <- train(quality ~ .,
                 data = wine.train,
                 method = "Rborist",
                 trControl = controls,
                 tuneGrid = data.frame(predFixed = 3,
                                       minNode = c(3, 50)),
                 preProcess = "scale",
                 nTree = 150)
summary(wine.rf)
wine.rf$bestTune

rf.preds <- factor(predict(wine.rf, wine.test),
                   levels = levels(wine.test$quality))

rf.cm <- confusionMatrix(rf.preds,
                         wine.test$quality)
rf.cm
