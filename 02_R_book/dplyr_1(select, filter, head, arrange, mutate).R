# R에서 '처리'부분에서 필요한 패키지 종류
# 데이터 소스 → 수집 → 저장 → 처리 → 분석 → 표현(빅데이터 전처리 과정)


# 데이터 처리
# dplyr(select/filter/head/arrange/mutate)

install.packages("dplyr")

library(dplyr)




# select(원하는 column을 추출하는 함수이다.)

exam %>%
  select(class, engilish)

ext_tracks %>%
  select(storm_name, month, day, hour, year, latitude)




# filter(원하는 row를 추출하는 함수이다.)

exam %>%
  filter(class == 1)

# 조건식 종류
# ==
# !=
# >
# >=
# <
# <=
# %in% # 인클루디드 인
# ex) storm_name %in% c("KATRINA", "ANDREW")
# = storm_name이 KATRINA나 ANDREW를 포함하고 있는지를 따져보는 연산자
# is.na()



ext_tracks %>%
  select(storm_name, hour, max_wind) %>%
  filter(hour == "00") %>%
  head(3)


ext_tracks %>%
  select(storm_name, month, day, hour, latitude, longitude, max_wind) %>%
  filter(storm_name == "ANDREW" & max_wind >= 137)




# arrange(순서대로 정렬하기)

exam %>%
  arrange(id) # id 오름차순(ascending)으로 정렬, asc을 써도 되지만 대부분 생략한다.

exam %>%
  arrange(desc(science)) # science 내림차순(descending)으로 정렬

exam %>%
  arrange(id, desc(science)) # id는 오름차순, science는 내림차순 정렬




# mutate(새로운 변수(컬럼) 추가하기)

exam %>%
  mutate(total = english + science)

exam %>%
  mutate(mean = total/2)
# mean 컬럼을 추가하고 english와 science의 평균을 넣어라

exam %>%
  mutate(test = ifelse(mean >= 60, "pass", "fail"))
# test 컬럼을 추가하고 mean이 60이상이면 "pass", 60 미만이면 "fail"로 마킹하라






