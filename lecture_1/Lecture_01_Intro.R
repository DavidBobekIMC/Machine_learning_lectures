# Working Directory ----
setwd("e:/OneDriveSiemens/FH-Krems/Lectures/")
getwd()
if (rstudioapi::isAvailable()) {setwd(dirname(rstudioapi::getActiveDocumentContext()$path))}

# Simple Math ----
1 + 2
'+'(1, 2) # (i.e. functional)
# handy as some libraries define new operators (e.g. %>%, %in%, %between%, etc.)

log(12)
log10(12)
sqrt(2)
2 ^ (1 / 2)
5 / 2 # division
5 %/% 2 # Quotient
5 %% 2 # Rest
# etc. etc.

# Assignment ----
a <- 15
a = 15 # Equivalent to "<-". I will tolerate it but try to avoid it
12 -> a

# Variable types ----
a <- TRUE           # Logical
b <- 3.15           # Numeric
c <- 3L             # Integer
d <- "Hello"        # Character
e <- 1 + 2i         # Complex
f <- factor("Red")  # Factor

## End of Lecture 1

class(a)
str(a)
# R does not require declaring a variable. The type is inferred depending on the value. 
# DYNAMICALLY TYPED LANGUAGE

# Data types ---- Vectors, Matrices (or Array), data.frames, and lists
# Vector
v <- c(25, 3, 18)
v[1]
v[2:3]
v[c(TRUE, TRUE, FALSE)]
v > 10
v[v > 10]

# Matrix
v <- 1:6
m <- matrix(v, ncol = 3, nrow = 2)
m[1, ]
m[, 2]
m[1, 2]

# Data.frame
df <- data.frame(x = 1:3, y = c("Mark", "John", "Laura"), z = c("m","m","f"))
# check structure
str(df)
# Select a column, both have the same result 
df[, "x"]
df$x

# Select a line
df[1, ]
# filter
df[df$z == "m", ]

# Lists
list <- list(
  Mark = c("gender" = "m", "income" = 20000),
  John = c("gender" = "m", "income" = 25000),
  Lara = c("gender" = "f", "income" = 28000))
# list can also contain different data types
list <- list(
  Period = "Quarter1",
  Orders = matrix(1:6, ncol = 3, nrow = 2),
  KPIs = list(growth = 0.2, Customers = 54))
list["Orders"] # returns a list
list[["Orders"]] # returns the raw element of the list
list[2] # can also use indexes

# If-condition ----
# if (condition) {
#   DO SOMETHING...
# } else {
#   DO SOMETHING ELSE...
# }
# ifelse(condition,{do_if_TRUE},{do_if_FALSE})

# For-loop ----
# VERY RARELY USED IN R, AND FOR A REASON!!!
# for(i in c(1:n)){
#   DO SOMETHING...
# }
# e.g.
# normal_dist[normal_dist>15]
# vs.
# for (i in 1:length(normal_dist)) {
#    if(normal_dist[i]>15) print(normal_dist[i])
# }
#
#


# Libraries ----
## Loading libraries (and libraries I always use) ----
library(data.table) # Super-fast data wrangling
library(dplyr) # Tidy-R. I am not a fan, but feel free to use
require(lubridate) # Date manipulation
library(plotly) # Wonderful plots
require(caret) # ML framework
library(mlr3) # ML framework
## Installing libraries ----
install.packages(c("mlr3", "caret"))
??mlr3 # help about a package
?data.table # help about a function
library(help="mlr3") # Info about the package
vignette("lubridate") # Vignette for a package

# misc ----
ls() # list the variable in the workspace
rm(a) # Remove one variable
rm(list = ls()) # clear the workspace
gc() # calls the garbage collector


normal_dist <- rnorm(1E3, 0, 10)
unif_dist <- runif(10, 0, 10,)
categories <- sample(LETTERS[1:3], size = 1E3, replace = T)
normal.dt <- data.table(x = categories, y = normal_dist, stringsAsFactors = T)
class(normal.dt)
str(normal.dt)
summary(normal.dt)
fivenum(normal.dt[, y])
head(normal.dt)
nrow(normal.dt)
ncol(normal.dt)

# *apply functions ----
lapply(normal.dt, fivenum) # apply fivenum to each column of normal.dt and returns a list
sapply(normal.dt, fivenum) # apply fivenum to each column of normal.dt and returns a data.frame
# vapply, mapply

# R-base vs. tidy-R vs. data.table----
## Example Grouping----
# R-base
aggregate(normal.dt$y, by = list(normal.dt$x), mean)
# Tidy-R
normal.dt %>% group_by(x) %>% summarise(Mean = mean(y), Median = median(y))
# Data.table
normal.dt[, .(Mean = mean(y), Median = median(y)), x] # dt[i, j, by] i.e., dt[where, select, group by]

## Example adding a column----
normal.dt <- normal.dt %>% mutate(y2 = y*2) # memcopy
normal.dt[, y2 := y * 2] # in-place addition