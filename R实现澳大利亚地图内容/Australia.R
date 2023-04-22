library(tidyverse)
library(stringr)
library(leaflet)
library(lubridate)
library(fuzzyjoin)
#Read two CSV data separately
#Count variables, 8 and 15 hours
#The longitude and latitude corresponding to the city in the data
#are combined with the Flight data Destination, 
#and the arrival time point Time_flight and morning 
#and night Day_or_night data are added to form a new table data storage
#Assign the location data of the storage to new_storage 
#and process the data into a list and create an array of the length of new_storage to new_total. 
#Use the while loop to calculate the number of times each location occurs 
#and assign the value to new_total. Finally, 
#assign the value of new_total to the newly created total column of storage.
#Storage1 was created to store data for Queensland and count the number of aircraft landings for each city. 
#The top four cities are calculated and the top four cities are assigned to forth_storage.
flight <- read_csv("RFDS_flightdata_July2022_PE2.csv")
basis <- read_csv("RFDS_bases_PE2.csv")
time_first = 8
time_second = 15
storage <-flight %>% 
  regex_inner_join(basis,by=c(Destination = 'Location')) %>% 
  mutate(Time_flight = hour(ArrivalAEST)) %>% 
  mutate(Day_or_night = if_else(between(Time_flight, time_first, time_first*2),"day_time","day_night"))
new_storage <- data.frame(storage$Location)
new_sum <- unlist(new_storage)
new_total <- array(length(new_sum))
i = 0
while(i< length(new_sum)+1){
  new_total[i] <- sum(new_sum[i] == storage$Location )
  i = i + 1
}
storage <- storage %>% 
  mutate(total = new_total)
storage1 <- storage %>% 
  group_by(Location,Day_or_night) %>% 
  count(name = "enumerate")
storage1 <- subset(storage1,Location %in% c("Roma","Brisbane","Bundaberg","Rockhampton","Charleville","Longreach","Mount Isa","Townsville","Cairns"))
storage1 %>%  
  group_by(Location) %>% 
  summarise(Total_enumerate = sum(enumerate))
forth_storage <- storage1 %>% 
  group_by(Location) %>% 
  summarise(Total_enumerate = sum(enumerate)) %>% 
  ungroup() %>% 
  slice_max(Total_enumerate, n = 4)
library(shiny)
Range_first = range(hour(storage$ArrivalAEST), na.rm = TRUE)
ui <- fixedPage(
  fixedRow(
    titlePanel(
      h2("Queensland Air Force base used by the RFDS, July 2022", align = "center")
      )
  ),
  fixedRow(
    substitute(
      ""
      ),
    tags$p("Data source RFDS flight data set in Australia", style="color: #8393E0;font-size:20px"), 
    tags$p("The number and duration of flights undertaken by RFDS aircraft in Queensland was studied",
           style="color: #8393E0;font-size:20px")
  ),
  fixedRow(
    column(4,
           tags$text("Base Location", style="color: #8393E0;font-weight:bold; font-size:20px;"),
           tags$br(),
           tags$p("According to the map data.
                   A total of 26 cities on the map have aircraft landing with RFDS, for a total of 821 landings.
                   The most RFDS aircraft landed at Brisbane with a total of 127 sorties, 
                   and the least at Richmond with a total of 1 sortie.
                   In picture, the northwest has the least number of flights and regions. 
                   Only Port Hedland Broome and Darwin have RFD flights, and the total number of flights in the three cities is 78.
                   In the northeastern state, the RFDS landed in the most cities and number of flights, with nine cities and 406 flights.
                   According to the picture, the RFDS planes land mainly in the three cities of Brisbane Jandakot and Adelaide. Not the megacities of Sydney and Melbourne.
                   According to the division of Australian administrative regions, 
                   Queensland is the state with the largest number and the largest number of aircraft landing with RFDS, 
                   while State of Victoria is the state with the smallest number and the least number of aircraft landing with RFDS.
                   The number of landings is 2, and the number of airports is 2, both in Melbourne.
                   The second lowest number of RFDS aircraft and landing cities is Northern Territory.
                   The total number of landings was 37, and the number of cities was 2, respectively in Darwin and Alice Springs.
                  "),
    ),
    column(8,
           sliderInput(
             "range",
             tags$p("The value ranges from 0 to 23:",
                    style="font-size:15px"),
             min = Range_first[1],
             max = Range_first[2],
             value = Range_first,
             width = '100%',
             round = TRUE,
             step = 1,
           ),
           leafletOutput("Australia_map"))
  ),
  fixedRow(
    column(
      width = 4,
      plotOutput("first")
    ),
    column(
      width = 3,
      tags$text("Time of arrival", style="color: #8393E0;font-weight:bold; font-size:16px;margin-left:50px"),
      tags$br(),
      tags$p("According to the aircraft arrivals chart for Queensland,
             nine cities in Queensland have RFDS landing,
             with Brisbane having the most flights, 127,
             and Longreach the least, with eight.
             There are almost the same number of flights during the day and night.
             According to the table of top four aircraft landing times in Queensland,
             it can be seen that among the four largest cities in Queensland,
             Brisbane has the highest number of aircraft landing times at 15:00 in the day,
             while there is no significant difference in the number of aircraft landing times between day and night in the other three cities.
             At night, few or no planes landed in the four cities during 0-6,
             while more planes landed at night during 16-23.")
    ),
    column(
      width = 5,
      plotOutput("second")
    )
  )
)

server <- function(input, output){
  output$second <- renderPlot({
    storage %>% 
      filter(Location %in% forth_storage$Location) %>% 
      group_by(Location,Time_flight,Day_or_night) %>% 
      count(name = "enumerate") %>% 
      ggplot(aes(x = Time_flight, y = enumerate, fill = Day_or_night))+
      facet_wrap(~Location)+
      scale_fill_manual(
        values = c(
          day_time = "#8393E0",
          day_night = "#211E20")
      )+
      theme_classic()+
      labs(
        title = "Table of top four aircraft landings in Queensland",
      )+
      theme(
            plot.title = element_text(
              face = "bold",
              hjust = 0.5,
              colour = "#8393E0",
              size = 16
            )
            )+
      geom_col(width = 0.66)
  })
  output$first <- renderPlot({
    ggplot(storage1) +
      aes(x = Location,y = enumerate, fill = Day_or_night)  +
      labs(
        title = "Aircraft Arrival Chart for Queensland"
      )+
      scale_fill_manual(
        values = c(
          day_time = "#8393E0",
          day_night = "#211E20")
      ) +
      coord_flip()+
      theme_classic()+
      geom_col(width = 0.65)+
      theme(plot.title=element_text(hjust=0.5))+
      theme(
        legend.position = c(0.79,0.79),
        plot.title = element_text(
          face = "bold",
          hjust = 0.3,
          colour = "#8393E0",
          size = 16
        )
      )
  })
  output$Australia_map <- renderLeaflet({
    storage %>% 
      filter(Time_flight >= input$range[1]) %>% 
      filter(Time_flight <= input$range[2]) %>% 
      group_by(Location,Latitude,Longitude,total) %>% 
      count(name = 'enumerate') %>% 
      leaflet() %>% 
      addTiles() %>% 
      addCircles(
        lng = ~Longitude,
        lat = ~Latitude,
        radius = ~enumerate*1999,
        popup = ~paste('The name of the place:', Location, "<br>Flight enumerate:",enumerate,"<br> Toatal Flight enumerate:", total )
      ) 
  })
}
shinyApp(ui,server)