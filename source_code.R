library(readr)

cardio.data <- read_delim("cardio_train.csv",
                           delim = ";", escape_double = FALSE, trim_ws = TRUE)

# Take all of the classification variables as their factors.
cardio.data$gender = as.factor(cardio.data$gender)
cardio.data$cholesterol = as.factor(cardio.data$cholesterol)
cardio.data$gluc = as.factor(cardio.data$gluc)
cardio.data$smoke = as.factor(cardio.data$smoke)
cardio.data$alco = as.factor(cardio.data$alco)
cardio.data$active = as.factor(cardio.data$active)
cardio.data$cardio = as.factor(cardio.data$cardio)

heart.logistic1 =  glm(cardio  ~  . - id,  family  =  "binomial",
                       data  =  cardio.data)

summary(heart.logistic1)

heart.logistic2 =  glm(cardio  ~  age+height+weight+ap_hi+ap_lo+cholesterol+smoke+alco+active
                       ,  family  =  "binomial",
                       data  =  cardio.data)

summary(heart.logistic2)

step(heart.logistic1)

heart.logistic3 = glm(formula = cardio ~ age + height + weight + ap_hi + ap_lo + 
                        cholesterol + gluc + smoke + alco + active, family = "binomial", 
                      data = cardio.data)
summary(heart.logistic3)

extract_info  = function(model) {
  deviance <- summary(model)$null.deviance
  residual_deviance <- summary(model)$deviance
  r_squared <- 1 - (residual_deviance / deviance)
  AIC <- AIC(model)
  BIC <- BIC(model)
  
  return(c(Null_Deviance = deviance,
           Residual_Deviance = residual_deviance,
           R_Squared = r_squared,
           AIC = AIC,
           BIC = BIC))
}

# Extract information from each model
info1 <- extract_info(heart.logistic1)
info2 <- extract_info(heart.logistic2)
info3 <- extract_info(heart.logistic3)

# Create a data frame to store the information
model_info <- data.frame(
  Model = c("heart.logistic1", "heart.logistic2", "heart.logistic3"),
  Null_Deviance = c(info1["Null_Deviance"], info2["Null_Deviance"], info3["Null_Deviance"]),
  Residual_Deviance = c(info1["Residual_Deviance"], info2["Residual_Deviance"], info3["Residual_Deviance"]),
  R_Squared = c(info1["R_Squared"], info2["R_Squared"], info3["R_Squared"]),
  AIC = c(info1["AIC"], info2["AIC"], info3["AIC"]),
  BIC = c(info1["BIC"], info2["BIC"], info3["BIC"])
)
(model_info)

set.seed(100)
test_errors = numeric(10)
for(i in 1:10){
  # initialize vector to store prediction errors
  sample= sample.int(n = nrow(cardio.data), size = floor(0.80*nrow(cardio.data)))
  train.heart.logistic = cardio.data[sample,]
  test.heart.logistic = cardio.data[-sample,]
  
  train.logistic = glm(formula = cardio ~ age + height + weight + ap_hi + ap_lo + 
                         cholesterol + gluc + smoke + alco + active, family = "binomial", 
                       data = cardio.data)
  
  glm.pred = predict.glm(train.logistic, newdata = test.heart.logistic, type = "response")
  
  pred = predict(train.logistic, type = "response", newdata = test.heart.logistic)
  val = ifelse(pred <0.5,"0", "1")
  tab = table(val, test.heart.logistic$cardio)
  test_errors[i] = (tab[2]+tab[3])/(tab[1]+tab[2]+tab[3]+tab[4])
}
(mean_test_error = mean(test_errors))

library(NeuralNetTools)
library(nnet)

set.seed(35)

# Separate the data into 80% training and 20% testing.
sample = sample(1 : nrow(cardio.data), floor(nrow(cardio.data) * 0.8))
cardio.train = cardio.data[sample, ]
cardio.test = cardio.data[-sample, ]

# Build the neural network.
cardio.model = nnet(cardio ~ . - id - gender, data = cardio.train,
                    size = 6, rang = 0.5, decay = 5e-2, maxit = 5000)

# Print the model's information, plot, summary, and garson plot.
print(cardio.model)
plotnet(cardio.model)
summary(cardio.model)
garson(cardio.model)

# Training prediction values.
cardio.train.predict = predict(cardio.model, cardio.train, type = "class")
cardio.train.values = cardio.train$cardio

# Testing prediction values.
cardio.test.predict = predict(cardio.model, cardio.test, type = "class")
cardio.test.values = cardio.test$cardio

# Training confusion matrix.
cardio.train.table = table(cardio.train.values, cardio.train.predict)
cardio.train.table

# Testing confusion matrix.
cardio.test.table = table(cardio.test.values, cardio.test.predict)
cardio.test.table