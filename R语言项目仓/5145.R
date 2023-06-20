library(tidyverse) 
library(dplyr)
library(stringr)
library(ggplot2)
library(patchwork)
library(plotrix)
options(scipen = 200)
csv_data <- read_csv('Wastes.csv')
#1
nrow(csv_data)
ncol(csv_data)
csv_data[1:10,]
csv_data[nrow(csv_data)-9:nrow(csv_data),]

#2
length(unique(csv_data$Type))
unique(csv_data$Type)
sum(grepl('chemicals',csv_data$Type)+grepl('organics',csv_data$Type))

#3
na_count <- 0
na_name <- c()
na_percent <- c()
for(i in 1:ncol(csv_data)){
  if(sum(is.na(csv_data[[i]])) > 0){
    na_count <- na_count + 1
    na_name <- append(na_name,i) 
    na_percent <- append(na_percent,sum(is.na(csv_data[[i]]))/length(csv_data[[i]]))
  }
}
print(na_count)
print(colnames(csv_data[na_name]))
sprintf('%1.8f%%',na_percent*100)

#4
sum(duplicated(csv_data$Case_ID))
repeat_data <- duplicated(csv_data$Case_ID)
repeat_list <- 0
for(i in 1:nrow(csv_data)){
  if(repeat_data[i] == TRUE){
    if(length(repeat_list) == 1){
      repeat_list <- csv_data[i,]
    }
    else{
      repeat_list <- rbind(repeat_list,csv_data[i,])
    }
  }
}
repeat_list
new_data <- filter(csv_data,!duplicated(csv_data$Case_ID))

#5
nrow(new_data)
table(new_data$`Core_Non-core`,useNA = "ifany")

#6
sum_data <- data.frame(new_data$Category,new_data$Tonnes)
names(sum_data) <- c('Category','Tonnes')
total_data <- aggregate(Tonnes ~ Category , data = sum_data,sum)
total_data
ggplot(data = total_data,mapping = aes(x = Category ,y=Tonnes))+ geom_bar(stat = 'identity') + geom_text(mapping = aes(label = Tonnes))

#7
Fate_tonnes <- data.frame(new_data$Fate,new_data$Tonnes)
names(Fate_tonnes) <- c('Fate','Tonnes')
Fate_tonnes <- aggregate(Tonnes ~ Fate , data = Fate_tonnes,sum)
Fate_tonnes <- arrange(Fate_tonnes,-Tonnes)
Fate_tonnes[1:2,]

#8
wastes_organic <- filter(new_data,Category == 'Organics')
organic_total <- data.frame(wastes_organic$Type,wastes_organic$Stream,wastes_organic$Tonnes)
names(organic_total) <- c('Type','Stream','Tonnes')
organic_total <- aggregate(Tonnes ~ Type + Stream,data = organic_total,sum)
organic_total
write.csv(x = organic_total,file = 'wastes_organics_type_stream.csv')
ggplot(data = organic_total,aes(Type,Tonnes,fill=Stream))+ geom_bar(stat = 'identity',position="dodge")

#9
year_data <- read_csv('Year_State_ID.csv')
names(year_data) <- c('Year_State_ID','Year','State')
new_list <- merge(new_data, year_data,by='Year_State_ID')
new_list[1:10,]

#10
State_tonnes <- data.frame(new_list$State,new_list$Tonnes)
names(State_tonnes)<- c('State','Tonnes')
State_unique <- unique(State_tonnes$State)
State_mean <- c()
State_min <- c()
State_max <- c()
for(i in State_unique){
  state_tonnes_cal <- State_tonnes %>%
    filter(State %in% i)
  state_tonnes_cal <- na.omit(state_tonnes_cal)
  State_min <- append(State_min,min(state_tonnes_cal$Tonnes))
  State_max <- append(State_max,max(state_tonnes_cal$Tonnes)) 
  State_mean <- append(State_mean,mean(state_tonnes_cal$Tonnes)) 
}
State_cal <- data.frame(State_unique,State_max,State_min,State_mean)
names(State_cal) <- c('State','Max','Min','Mean')
State_cal <- arrange(State_cal,-Mean)
State_cal
sprintf('The states with the highest mean are: %s', State_cal$State[1])
p = ggplot(data = State_cal,aes(x = reorder(State,-Mean) ,y = Mean ))+ geom_bar(stat = 'identity') 
p + xlab("Sate") + ylab("Mean") + ggtitle("Ranking by the average of tonnes in each state")

#11
Recycle_tonnes <- new_list %>% 
  filter(Fate %in% 'Recycling')
Recycle_tonnes <- data.frame(Recycle_tonnes$Type,Recycle_tonnes$Tonnes,Recycle_tonnes$Year)
names(Recycle_tonnes) <- c('Type','Tonnes','Year')
Recycle_tonnes <- aggregate(Tonnes ~ Type + Year,data = Recycle_tonnes,sum)
Disposal_tonnes <- new_list %>% 
  filter(Fate %in% 'Disposal')
Disposal_tonnes <- data.frame(Disposal_tonnes$Type,Disposal_tonnes$Tonnes,Disposal_tonnes$Year)
names(Disposal_tonnes) <- c('Type','Tonnes','Year')
Disposal_tonnes <- aggregate(Tonnes ~ Type + Year,data = Disposal_tonnes,sum)
Waste_reuse_addtonnes <- new_list %>% 
  filter(Fate %in% 'Waste reuse')
Waste_reuse_addtonnes <- data.frame(Waste_reuse_addtonnes$Type,Waste_reuse_addtonnes$Tonnes,Waste_reuse_addtonnes$Year)
names(Waste_reuse_addtonnes) <- c('Type','Tonnes','Year')
Waste_reuse_addtonnes <- aggregate(Tonnes ~ Type + Year,data = Waste_reuse_addtonnes,sum)
Waste_reuse_addtonnes <- Waste_reuse_addtonnes %>% select(Year,Type,Tonnes)
Waste_reuse_addtonnes <- arrange(Waste_reuse_addtonnes,Type,Year)
add_waste <- c()
for(i in 2:nrow(Waste_reuse_addtonnes)){
  if(Waste_reuse_addtonnes[i-1,2] == Waste_reuse_addtonnes[i,2]){
    add_waste <- append(add_waste,Waste_reuse_addtonnes[i,3]-Waste_reuse_addtonnes[i-1,3])
  }
  else{
    add_waste <- append(add_waste,0)
  }
}
add_waste<-append(add_waste,0)
Waste_reuse_addtonnes <- data.frame(Waste_reuse_addtonnes$Type,add_waste)
names(Waste_reuse_addtonnes) <- c('Type','Add_value')
Waste_reuse_addtonnes <- aggregate(Add_value ~ Type ,data = Waste_reuse_addtonnes,sum)
Waste_reuse_addtonnes <- arrange(Waste_reuse_addtonnes,-Add_value,Type)
Recycle_tonnes[which.max(Recycle_tonnes$Tonnes),]
Disposal_tonnes[which.max(Disposal_tonnes$Tonnes),]
sprintf('The Type with the largest increase is: %s The value of the increase is: %f',Waste_reuse_addtonnes[1,1],Waste_reuse_addtonnes[1,2])

#12
Hazardous_wastes <- new_list %>% 
  filter(Category %in% 'Hazardous wastes') %>% 
  filter(Type %in% 'Tyres (T140)') %>% 
  filter(Tonnes > 0)
Tonnes_range <- c()
for (i in Hazardous_wastes$Tonnes) {
  if(i<10000){
    Tonnes_range <- append(Tonnes_range,'[0,10000)')
  }
  else if(i>=10000 && i < 20000 ){
    Tonnes_range <- append(Tonnes_range,'[10000,20000)')
  }
  else if(i>=20000 && i < 40000 ){
    Tonnes_range <- append(Tonnes_range,'[20000,40000)')
  }
  else if(i>=40000 && i <= 80000 ){
    Tonnes_range <- append(Tonnes_range,'[40000,80000]')
  }
}
Hazardous_wastes <- data.frame(Hazardous_wastes,Tonnes_range)
ggplot(data = Hazardous_wastes,aes(State,Tonnes_range,fill=Year))+ geom_bar(stat = 'identity',position="dodge")
View(Hazardous_wastes)

#13
state_waste_food <- new_list %>% 
  filter(Type %in% 'Food organics')
for (i in 1:nrow(state_waste_food)) {
  state_waste_food[i,9] <- substr(state_waste_food[i,9],1,4)
}
state_waste_food <- data.frame(state_waste_food$Year,state_waste_food$Tonnes,state_waste_food$State)
names(state_waste_food) <- c('Year','Tonnes','State')
state_waste_food <- aggregate(Tonnes ~ Year + State ,data = state_waste_food,sum)
ggplot(data = state_waste_food,aes(x=Year,y=Tonnes,group = State,color=State,shape=State))+
  geom_point()+
  geom_line()+
  xlab("Year")+
  ylab("Value")+
  theme_bw()

#14
CD_list <- new_list %>% 
  filter(Stream %in% 'C&D')
CD_list <- data.frame(CD_list$Year,CD_list$Tonnes)
for (i in 1:nrow(CD_list)) {
  CD_list[i,1] <- substr(CD_list[i,1],1,4)
}
CD_list <- aggregate(CD_list.Tonnes ~ CD_list.Year,data = CD_list,sum)
names(CD_list) <- c('Year','C&D')
AUS_mining <- read_csv('australia-mining.csv')
#source:https://www.abs.gov.au/statistics/labour/employment-and-unemployment/labour-force-australia-detailed/may-2022
AUS_mining_factor <- data.frame(AUS_mining$Year,AUS_mining$Number_mining)
names(AUS_mining_factor) <- c('Year','Factor:Number_mining') 
CD_factor <- merge(CD_list, AUS_mining_factor,by.x='Year',by.y = 'Year')
View(CD_factor)
plot(x = CD_factor$`C&D`,y = CD_factor$`Factor:Number_mining`,
     xlab = "C&D",
     ylab = "Factor:Number_mining",
     main = "The relationship between mining and C&D "
)

#15
state_Energy_recovery <- new_list %>% 
  filter(Fate %in% 'Energy recovery') %>% 
  filter(Tonnes > 0)
state_Energy_recovery$Year <- substr(state_Energy_recovery$Year,1,4)
state_Energy_recovery <- data.frame(state_Energy_recovery$Year,state_Energy_recovery$Type,state_Energy_recovery$Tonnes)
names(state_Energy_recovery) <- c('Year','Type','Tonnes')
state_Energy_recovery <- aggregate(Tonnes ~ Year + Type ,data = state_Energy_recovery,sum)
p1 <- ggplot(state_Energy_recovery)+
    geom_bar(aes(x=Year,y=Tonnes,fill=Type), stat = 'identity',position="dodge")
p2<- ggplot(state_Energy_recovery)+
  geom_line(aes(x=Year,y=Tonnes,group = Type,color=Type))+
  xlab("Year")+
  ylab("Value")+
  theme_bw()
energy_tonnes <- data.frame(state_Energy_recovery$Type,state_Energy_recovery$Tonnes)
names(energy_tonnes) <- c('Type','Tonnes')
energy_tonnes <- aggregate(Tonnes ~ Type,data =energy_tonnes ,sum)
energy_tonnes <- arrange(energy_tonnes,-Tonnes)
account <- 0
for(i in 7:nrow(energy_tonnes)){
  account <- account+energy_tonnes[i,2]
}
energy_tonnes <- energy_tonnes[-c(7,8,9),]
energy_tonnes[6,2] <- energy_tonnes[6,2] + account
p1+p2
piepercent = paste(round(100*energy_tonnes$Tonnes/sum(energy_tonnes$Tonnes)), "%")
piepercent_name <- c()
for(i in 1:length(piepercent)){
  print(class(energy_tonnes[i,1]))
  print(class(piepercent[i]))
  piepercent_name <- append(piepercent_name,paste(energy_tonnes[i,1],piepercent[i]))
}
pie3D(energy_tonnes$Tonnes,labels =piepercent_name,explode = 0.1, main = "The proportion of Energy recovery by different categories in Australia")


