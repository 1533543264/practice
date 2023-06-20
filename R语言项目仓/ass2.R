library(tidyverse) 
library(dplyr)
library(stringr)
library(ggplot2)
library(patchwork)
library(plotrix)
library(gdata)
library(readxl)
library(reshape2)
library(caret)
library(randomForest)
library(rnaturalearth)
#引入xlsx文件
path <- 'datascience.xlsx'
#读取全部的表单信息
df <- excel_sheets(path) %>% 
  set_names() %>% 
  map(read_excel, path=path)
sheet_names <- excel_sheets(path)
# 创建一个空的列表
sheet_data <- vector("list", length = length(sheet_names))
sheet<-list()
# 遍历每个工作表
for (i in 1:length(sheet_names)) {
  sheet_name <- sheet_names[i]
  data <- read_excel(path, sheet = sheet_name)
  # 存储数据到列表中
  sheet <- append(sheet, list(data))
}
has_na <- list()

for (i in 1:length(sheet)) {
  sheet_dataframe <- do.call(data.frame, sheet[i])
  has_na <- any(is.na(sheet_dataframe))
  if (has_na) {
    # 创建一个新的数据框，其中只包含含有缺失值的列
    na_columns <- colnames(sheet_dataframe)[colSums(is.na(sheet_dataframe)) > 0]
    na_data <- sheet_dataframe[, na_columns]
    
    # 检查是否存在符合条件的数据行
    if (nrow(na_data) > 0) {
      # 将数据从宽格式转换为长格式
      na_data_long <- tidyr::pivot_longer(na_data, cols = everything(), names_to = "variable", values_to = "value")
      # 将变量名转换为字符向量
      na_data_long$variable <- as.character(na_data_long$variable)
      # 提取每个项目的前三个字母
      na_data_long$variable_short <- ifelse(nchar(na_data_long$variable) > 5, substr(na_data_long$variable, start = 1, stop = 5), na_data_long$variable)
      
      # 创建箱型图
      print(ggplot(na_data_long, aes(x = variable_short, y = value)) +
              geom_boxplot() +
              labs(title = "Boxplot of NA Values") +
              xlab("Variable") +
              ylab("Value"))
    } else {
      print("No rows with missing values.")
    }
  } else {
    print("No missing values in the dataset.")
  }
}



#根据对数据的分析，可以将NA值设置为0
#因为数据采用百分比统计，我认为可以将NA值设置为0
# 将所有NA值设置为0
for (i in 1:length(sheet)) {
  sheet_dataframe <- do.call(data.frame, sheet[i])
  sheet_dataframe[is.na(sheet_dataframe)] <- 0
  sheet[i] <- list(sheet_dataframe)
}
#证明已经修改成功
for (i in 1:length(sheet)) {
  sheet_dataframe <- do.call(data.frame, sheet[i])
  has_na <- any(is.na(sheet_dataframe))
  print(has_na)
}



newdata <-  rbind(sheet[[3]],sheet[[3]])
newdata <- rbind(newdata,sheet[[3]])
newdata <- rbind(newdata,sheet[[3]])
newdata <- rbind(newdata,sheet[[3]])
# 假设您已经准备好了训练数据 trainData 和目标变量 trainLabels
# 定义新的 mtry 取值
new_mtry <- 10
data(newdata )
# 划分数据集为训练集和测试集
set.seed(123)
trainIndex <- createDataPartition(newdata$State.Territory, p = 0.7, list = FALSE)
trainData <- newdata[trainIndex, ]
testData <- newdata[-trainIndex, ]
trainData$State.Territory <- as.factor(trainData$State.Territory)
# 训练机器学习模型（这里以随机森林算法为例）
# 设置 randomForest 函数的 mtry 参数为新的取值
model <- randomForest(State.Territory ~ .,trainData, mtry = new_mtry)


# 在测试集上进行预测
predictions <- predict(model, newdata = testData)
# 将预测结果和测试数据的目标变量转换为因子类型
predictions <- factor(predictions)
testData$State.Territory <- factor(testData$State.Territory)
# 创建混淆矩阵
cm <- confusionMatrix(predictions, testData$State.Territory)

# 绘制混淆矩阵热力图
heatmap(cm$table, Colv = NA, Rowv = NA, 
        col = colorRampPalette(c("white", "darkgreen"))(50),
        main = "Confusion Matrix Heatmap", xlab = "Predicted", ylab = "Actual")

# 绘制预测结果与真实标签的散点图
ggplot(data = data.frame(actual = testData$State.Territory, predicted = predictions)) +
  geom_point(aes(x = predicted, y = actual)) +
  labs(title = "Scatter plot of Predicted vs Actual", x = "Predicted", y = "Actual")


# 数据准备
sales <- sheet[[3]]$Income.from.sales.of.goods.or.services....Stayed.the.same.since.last.year # 销售额
environment <- sheet[[3]]$Environmental.focus....Stayed.the.same.since.last.year # 创新环境

#描述性统计分析
# 计算平均值
mean_sales <- mean(sales)

# 计算标准差
sd_sales <- sd(sales)

# 计算最小值
min_sales <- min(sales)

# 计算最大值
max_sales <- max(sales)

# 计算分位数
quantiles_sales <- quantile(sales, probs = c(0.25, 0.5, 0.75))
sales_data <- data.frame(sales) 
# 绘制直方图
ggplot(sales_data, aes(x = sales)) +
  geom_histogram(binwidth = 20, fill = "steelblue", color = "white") +
  labs(x = "销售额", y = "频数", title = "销售额分布直方图")

# 绘制箱线图
ggplot(sales_data, aes(y = sales)) +
  geom_boxplot(fill = "lightblue", color = "steelblue") +
  labs(y = "销售额", title = "销售额箱线图")

# 绘制密度图
ggplot(sales_data, aes(x = sales)) +
  geom_density(fill = "lightblue", color = "steelblue") +
  labs(x = "销售额", y = "密度", title = "销售额密度图")

#聚类分析
# 数据准备
company_data <- data.frame(sales, environment) # 创建数据框
# 执行聚类分析
k <- 3 # 指定聚类的数量
kmeans_model <- kmeans(company_data, centers = k)

# 查看聚类结果
cluster_labels <- kmeans_model$cluster
cluster_labels
# 绘制散点图，并按聚类标签着色
ggplot(company_data, aes(x = sales, y = rd_investment, color = factor(cluster_labels))) +
  geom_point(size = 4) +
  labs(x = "销售额", y = "研发投入", color = "聚类标签") +
  scale_color_discrete(name = "聚类") +
  theme_minimal()


#数据之间回归模型预测
innovation_data <- data.frame(sales, environment) # 创建数据框

# 构建回归模型
model <- lm(sales ~ environment, data = innovation_data)

# 查看回归模型结果
summary(model)

# 预测销售额
new_environment <- data.frame(environment = 100)
predicted_sales <- predict(model, newdata = new_environment)
predicted_sales





# 获取澳大利亚地图数据
aus_map <- ne_states(country = "Australia", returnclass = "sf")

# 选择要显示的州
states_to_show <- c("New South Wales", "Queensland", "South Australia", "Tasmania", "Victoria", "Western Australia", "Australian Capital Territory", "Northern Territory")
aus_map_filtered <- subset(aus_map, name %in% states_to_show)

# 创建地图图表
map_plot <- ggplot() +
  geom_sf(data = aus_map_filtered, aes(fill = name), color = "black") +
  coord_sf() +
  theme_void()

# 设置各州的颜色
state_colors <- c("New South Wales" = "#084594",
                  "Victoria" = "#2171b5",
                  "Queensland" = "#4292c6",
                  "South Australia" = "#6baed6",
                  "Western Australia" = "#9ecae1",
                  "Tasmania" = "#c6dbef",
                  "Northern Territory" = "#eff3ff")

# 创建每个州的数据
state_data <- data.frame(name = states_to_show,
                         value = c(354, 190, 56, 17, 260, 90, 0, 5))

# 合并州的几何信息和数据
aus_map_data <- merge(aus_map_filtered, state_data, by = "name")

# 在地图上添加州的标签和数值标签
map_plot <- map_plot +
  geom_sf_text(data = aus_map_data, aes(label = value), color = "black", size =3, nudge_y = -0.5) +
  scale_fill_manual(values = state_colors)

# 获取各州的中心坐标
center_data <- data.frame(state = c("NSW", "QLD", "SA", "TAS", "VIC", "WA", "ACT", "NT"),
                          lon = c(146.8, 143.1, 135.0, 147.0, 144.8, 121.3, 150.1, 133.2),
                          lat = c(-31.7, -21.9, -30.5, -42.0, -37.8, -27.5, -35.3, -20.8))
state_data$name
# 添加州名
map_plot <- map_plot +
  geom_text(data = center_data, aes(x = lon, y = lat, label = state), color = "black", size = 2.5)
map_plot <- map_plot +
  labs(fill = "state")  
map_plot 
