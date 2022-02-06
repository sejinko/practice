# R에서 '처리'부분에서 필요한 패키지 종류
# 데이터 소스 → 수집 → 저장 → 처리 → 분석 → 표현(빅데이터 전처리 과정)


# 데이터 처리
# dplyr(summarize/group_by)

library(dplyr)




# summarise

ext_tracks %>%
  summarise(n_obs = n(), # The number of observations(관찰 개수)
            worst_wind = max(max_wind), # 최대값
            worst_pressure = min(min_pressure)) # 최소값




# group_by()

ext_tracks %>%
  group_by(storm_name, year) %>%
  head()




# Groiping / Summarizing Combination(두개를 같이 사용하면 강력한 효과를 냄)

ext_tracks %>%
  group_by(strom_name, year) %>%
  summarize(n_obs = n(),
            worst_wind = max(max_wind),
            worst_pressure = min(min_pressure))



knots_to_mph <- function(knots){
  mph <- 1.152 * knots
} # 사용자 정의 함수를 써서 단위변경 함수를 만들고 아래처럼 사용한다.

ext_tracks %>%
  summarize(n_obs = n(),
            worst_wind = knots_to_mph(max(max_wind)),
            worst_pressure = min(min_pressure))


# 함수 구조

함수명 <- function(arg1, arg2, arg3, ...){
  로직(=알고리즘)
}

