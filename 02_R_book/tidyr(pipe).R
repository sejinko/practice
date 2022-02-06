# R에서 '처리'부분에서 필요한 패키지 종류
# 데이터 소스 → 수집 → 저장 → 처리 → 분석 → 표현(빅데이터 전처리 과정)
# dplyr, tidyr, stringr, lubridate

# tidyr(pipe)


입력값 %>% 결과값
# 함수의 결과값이 다른 함수의 입력값으로 바로 들어가게 할때 사용한다.

# without piping
# function(dataframe, argument_2, argument_3)

# with piping
# dataframe %>%
#  function(argument_2, argument_3)



install.packages("tidyr")

library(tidyr)

ext_tracks %>%
  filter(storm_name == "KATRINA") %>%
  select(month, day, hour, max_wind) %>%
  head(3)

# pipe(%>%)를 사용하지 않으면 아래와 같이 사용해야 한다.
# 1번째(메모리가 낭비됨)
katrina <- filter(ext_tracks, storm_name == "KATRINA")
katrina_reduced <- select(katrina, month, day, hour, max_wind)
head(katrina_reduced, 3)

# 2번째(메모리가 낭비되지는 않지만 해석하기 어려움)
head(select(filter(ext_tracks, storm_name == "KATRINA"),
            month, day, hour, max_wind), 3)





















