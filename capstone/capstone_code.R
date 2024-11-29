##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes
# install and load relevant libraries

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(Hmisc)) install.packages("Hmisc", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dlabs", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(dslabs)
ds_theme_set()

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# view summary of edx data set
class(edx)
summary(edx)
head(edx)

set.seed(1984)

# recode variables and view summary
edx.mutated <- edx %>% 
  mutate(year = factor(str_match(edx$title, "\\((\\d{4})\\)$")[,2]),
         date = as_datetime(timestamp),
         userId = factor(userId),
         movieId = factor(movieId),
         genres = factor(str_match(edx$genres, "^[^|]+")))
summary(edx.mutated)
head(edx.mutated)

# partition testing and training sets
test.index <- createDataPartition(edx.mutated$rating, 
                                  p = 0.2, 
                                  list = F)
edx.test <- edx.mutated[test.index, ]
edx.train <- edx.mutated[-test.index, ]

# ensure that all movies in training set are also in testing set
edx.test <- edx.test %>% 
  semi_join(edx.train, by = "userId")
edx.train <- edx.train %>% 
  semi_join(edx.test, by = "userId")

# correlation matrix
edx.train %>% 
  select(-c(rating, genres, title, date)) %>% 
  as.matrix() %>% 
  Hmisc::rcorr(type = "spearman")

# examine counts, average ratings, and range by movie
edx.train %>% 
  group_by(movieId) %>% 
  summarise(movie.avg = mean(rating),
            rating.count = n()) %>% 
  summarise(range = max(rating.count) - min(rating.count))

# histogram of the distribution of rating counts among movies
edx.train %>% 
  group_by(movieId) %>% 
  summarise(movie.avg = mean(rating),
            rating.count = n()) %>%
  ggplot(aes(x = rating.count)) +
  geom_histogram(binwidth = 1000) +
  scale_y_log10()

# examine the distribution of ratings by user
edx.train %>% 
  group_by(movieId) %>% 
  summarise(user.avg = mean(rating),
            user.sd = sd(rating),
            rating.count = n()) %>% 
  arrange(desc(rating.count))

edx.train %>% 
  group_by(userId) %>% 
  summarise(user.avg = mean(rating),
            rating.count = n()) %>% 
  summarise(range = max(rating.count) - min(rating.count))

# histogram of the distribution of rating counts among users
edx.train %>% 
  group_by(userId) %>% 
  summarise(user.avg = mean(rating),
            ratings.count = n()) %>%
  ggplot(aes(x = ratings.count)) +
  geom_histogram(binwidth = 100) +
  scale_y_log10()

# examine counts, average ratings, and range by genre
edx.train %>% 
  group_by(genres) %>% 
  summarise(genre.avg = mean(rating),
            count = n()) %>% 
  arrange(desc(count))

edx.train %>% 
  group_by(genres) %>% 
  summarise(genre.avg = mean(rating),
            count = n()) %>% 
  arrange(desc(genre.avg))

edx.train %>% 
  group_by(genres) %>% 
  summarise(genre.avg = mean(rating),
            rating.count = n()) %>% 
  summarise(range = max(rating.count) - min(rating.count))

# bar graph of rating counts by genre
edx.train %>% 
  group_by(genres) %>% 
  summarise(genre.avg = mean(rating),
            rating.count = n()) %>%
  ggplot(aes(x = reorder(genres, desc(rating.count)), y = rating.count)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  labs(title = "Rating Counts by Genre") +
  xlab("Genre") +
  ylab("Count of Ratings")

# counts, average ratings, and range by year
edx.train %>% 
  group_by(year) %>% 
  summarise(year.avg = mean(rating),
            rating.count = n()) %>% 
  arrange(desc(year.avg))

edx.train %>% 
  group_by(year) %>% 
  summarise(year.avg = mean(rating),
            rating.count = n()) %>% 
  arrange(desc(rating.count))

edx.train %>% 
  group_by(year) %>% 
  summarise(year.avg = mean(rating),
            rating.count = n()) %>% 
  summarise(range = max(rating.count) - min(rating.count))

# bar graph of rating counts by year
edx.train %>% 
  group_by(year) %>% 
  summarise(year.avg = mean(rating),
            rating.count = n()) %>%
  ggplot(aes(x = year, y = rating.count)) +
  geom_bar(stat = "identity") +
  scale_x_discrete(breaks = seq(1915, 2008, 5)) +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  labs(title = "Rating Counts by Year") +
  xlab("Year") +
  ylab("Count of Ratings")

# regularization procedures: define overall average `mu` and bias penalty ranges to test
mu <- mean(edx.train$rating)
lambdas <- seq(0, 20, 0.1)
more.lambdas <- seq(0, 100, 5)
even.more.lambdas <- 10^seq(-2, 2, 0.1)

# find ideal lambda for movie regularization
movie.rmses <- sapply(even.more.lambdas, function(x) {
  edx.train %>% 
    mutate(mu = mean(rating),
           y.mu.diff = rating - mu) %>% 
    group_by(movieId) %>% 
    summarise(n.movies = n(),
              sum.movie.diff = sum(y.mu.diff),
              b_lambda = sum.movie.diff / (n.movies + x)) %>% 
    left_join(edx.test, by = "movieId") %>% 
    mutate(pred = mu + b_lambda) %>% 
    filter(!is.na(rating)) %>% 
    summarise(rmse = RMSE(pred, rating)) %>% 
    pull(rmse)
})

# cbind(lambdas, movie.rmses)
# cbind(more.lambdas, movie.rmses)
cbind(even.more.lambdas, movie.rmses)
l.movie <- even.more.lambdas[which.min(movie.rmses)]

# movie regularization
edx.train %>% 
  mutate(mu = mean(rating),
         y.mu.diff = rating - mu) %>% 
  group_by(movieId) %>% 
  summarise(n.movies = n(),
            sum.movie.diff = sum(y.mu.diff),
            b_lambda = sum.movie.diff / (n.movies + l.movie)) %>% 
  ungroup() %>% 
  left_join(edx.test, by = "movieId") %>% 
  mutate(pred = mu + b_lambda) %>% 
  filter(!is.na(rating)) %>%
  summarise(rmse = RMSE(pred, rating))

b.movie <- edx.train %>% 
  mutate(mu = mean(rating),
         y.mu.diff = rating - mu) %>% 
  group_by(movieId) %>% 
  summarise(n.movies = n(),
            sum.movie.diff = sum(y.mu.diff),
            b_movie = sum.movie.diff / (n.movies + l.movie)) %>% 
  ungroup()

# find ideal lambda for user regularization
user.rmses <- sapply(even.more.lambdas, function(x) {
  edx.train %>% 
    left_join(b.movie, by = "movieId") %>% 
    mutate(mu = mean(rating),
           y.mu.diff = rating - mu - b_movie) %>% 
    group_by(userId) %>% 
    summarise(n.users = n(),
              sum.user.diff = sum(y.mu.diff),
              b_lambda = sum.user.diff / (n.users + x)) %>% 
    left_join(edx.test, by = "userId") %>% 
    mutate(pred = mu + b_lambda) %>% 
    filter(!is.na(rating)) %>% 
    summarise(rmse = RMSE(pred, rating)) %>% 
    pull(rmse)
})

cbind(even.more.lambdas, user.rmses)
l.user <- even.more.lambdas[which.min(user.rmses)]

# user regularization
edx.train %>% 
  left_join(b.movie, by = "movieId") %>% 
  mutate(mu = mean(rating),
         y.mu.diff = rating - mu - b_movie) %>% 
  group_by(userId) %>%
  summarise(n.users = n(),
            sum.user.diff = sum(y.mu.diff),
            b_lambda = sum.user.diff / (n.users + l.user)) %>%
  left_join(edx.test, by = "userId") %>%
  mutate(pred = mu + b_lambda) %>%
  filter(!is.na(rating)) %>%
  summarise(RMSE(pred, rating))


b.user <- edx.train %>% 
  left_join(b.movie, by = "movieId") %>% 
  mutate(mu = mean(rating),
         y.mu.diff = rating - mu - b_movie) %>% 
  group_by(userId) %>% 
  summarise(n.users = n(),
            sum.user.diff = sum(y.mu.diff),
            b_user = sum.user.diff / (n.users + l.user)) %>% 
  ungroup()

# find ideal lambda for genre regularization
genre.rmses <- sapply(even.more.lambdas, function(x) {
  edx.train %>% 
    left_join(b.movie, by = "movieId") %>% 
    left_join(b.user, by = "userId") %>% 
    mutate(mu = mean(rating),
           y.mu.diff = rating - mu - b_movie - b_user) %>% 
    group_by(genres) %>% 
    summarise(n.genres = n(),
              sum.genre.diff = sum(y.mu.diff),
              b_lambda = sum.genre.diff / (n.genres + x)) %>% 
    left_join(edx.test, by = "genres") %>% 
    mutate(pred = mu + b_lambda) %>% 
    filter(!is.na(rating)) %>% 
    summarise(rmse = RMSE(pred, rating)) %>% 
    pull(rmse)
})

cbind(even.more.lambdas, genre.rmses)
l.genre <- even.more.lambdas[which.min(genre.rmses)]

# genre regularization
edx.train %>% 
  left_join(b.movie, by = movieId) %>% 
  left_join(b.user, by = userId) %>% 
  mutate(mu = mean(rating),
         y.mu.diff = rating - mu - b_movie - b_user) %>% 
  group_by(genres) %>% 
  summarise(n.genres = n(),
            sum.genre.diff = sum(y.mu.diff),
            b_lambda = sum.genre.diff / (n.genres + l.genre)) %>% 
  left_join(edx.test, by = "genres") %>% 
  mutate(pred = mu + b_lambda) %>% 
  filter(!is.na(rating)) %>% 
  summarise(RMSE(pred, rating))

b.genre <- edx.train %>% 
  left_join(b.movie, by = "movieId") %>% 
  left_join(b.user, by = "userId") %>% 
  mutate(mu = mean(rating),
         y.mu.diff = rating - mu - b_movie - b_user) %>% 
  group_by(genres) %>% 
  summarise(n.genres = n(),
            sum.genre.diff = sum(y.mu.diff),
            b_genre = sum.genre.diff / (n.genres + l.genre)) %>% 
  ungroup()

# find ideal lambda for year regularization
year.rmses <- sapply(even.more.lambdas, function(x) {
  edx.train %>% 
    left_join(b.movie, by = "movieId") %>% 
    left_join(b.user, by = "userId") %>% 
    left_join(b.genre, by = "genres") %>% 
    mutate(mu = mean(rating),
           y.mu.diff = rating - mu - b_movie - b_user - b_genre) %>% 
    group_by(year) %>% 
    summarise(n.year = n(),
              sum.year.diff = sum(y.mu.diff),
              b_lambda = sum.year.diff / (n.year + x)) %>% 
    left_join(edx.test, by = "year") %>% 
    mutate(pred = mu + b_lambda) %>% 
    filter(!is.na(rating)) %>% 
    summarise(rmse = RMSE(pred, rating)) %>% 
    pull(rmse)
})

cbind(even.more.lambdas, year.rmses)
l.year <- even.more.lambdas[which.min(year.rmses)]

# year regularization
edx.train %>% 
  left_join(b.movie, by = "movieId") %>% 
  left_join(b.user, by = "userId") %>%
  left_join(b.genre, by = "genres") %>% 
  mutate(mu = mean(rating),
         y.mu.diff = rating - mu - b_movie - b_user - b_genre) %>% 
  group_by(year) %>% 
  summarise(n.year = n(),
            sum.year.diff = sum(y.mu.diff),
            b_year = sum.year.diff / (n.year + l.year)) %>% 
  left_join(edx.test, by = "year") %>% 
  mutate(pred = mu + b_year) %>% 
  filter(!is.na(rating)) %>% 
  summarise(RMSE(pred, rating))

b.year <-edx.train %>% 
  left_join(b.movie, by = "movieId") %>% 
  left_join(b.user, by = "userId") %>%
  left_join(b.genre, by = "genres") %>% 
  mutate(mu = mean(rating),
         y.mu.diff = rating - mu - b_movie - b_user - b_genre) %>% 
  group_by(year) %>% 
  summarise(n.year = n(),
            sum.year.diff = sum(y.mu.diff),
            b_year = sum.year.diff / (n.year + l.year)) %>% 
  ungroup()

# model evaluation

# test naive model ("guessing")
guesses <- sample(x = seq(0.5, 5, 0.5),
                  size = nrow(edx.test),
                  replace = T)
RMSE(guesses, edx.test$rating)

# test movie bias model
edx.test %>% 
  left_join(b.movie, by = "movieId") %>% 
  mutate(pred = mu + b_movie) %>% 
  filter(!is.na(pred)) %>% 
  summarise(rmse = RMSE(rating, pred))

# test movie + user bias model
edx.test %>% 
  left_join(b.movie, by = "movieId") %>% 
  left_join(b.user, by = "userId") %>%
  mutate(pred = mu + b_movie + b_user) %>% 
  filter(!is.na(pred)) %>% 
  summarise(rmse = RMSE(rating, pred))

# test movie + user + genre bias model
edx.test %>% 
  left_join(b.genre, by = "genres") %>% 
  left_join(b.movie, by = "movieId") %>% 
  left_join(b.user, by = "userId") %>% 
  mutate(pred = mu + b_movie + b_user + b_genre) %>% 
  filter(!is.na(pred)) %>% 
  summarise(rmse = RMSE(rating, pred))

# test movie + user + genre + year bias model
edx.test %>% 
  left_join(b.year, by = "year") %>% 
  left_join(b.genre, by = "genres") %>% 
  left_join(b.movie, by = "movieId") %>% 
  left_join(b.user, by = "userId") %>% 
  mutate(pred = mu + b_movie + b_user + b_genre + b_year) %>% 
  filter(!is.na(pred)) %>% 
  summarise(rmse = RMSE(rating, pred))

# final testing
final_holdout_test %>% 
  mutate(year = factor(str_match(final_holdout_test$title, "\\((\\d{4})\\)$")[,2]),
         genres = factor(str_match(final_holdout_test$genres, "^[^|]+")),
         userId = factor(userId),
         movieId = factor(movieId)) %>% 
  left_join(b.year, by = "year") %>% 
  left_join(b.genre, by = "genres") %>% 
  left_join(b.movie, by = "movieId") %>% 
  left_join(b.user, by = "userId") %>%
  mutate(pred = mu + b_movie + b_user + b_genre + b_year) %>% 
  filter(!is.na(pred)) %>% 
  summarise(rmse = RMSE(rating, pred))