# R에서 '수집'부분에서 필요한 패키지 종류
# 데이터 소스 → 수집 → 저장 → 처리 → 분석 → 표현(빅데이터 전처리 과정)
# readr, rvest, xml2, httr, Rcrawler, Rselenium

#readr 패키지

install.packages("readr")

library(readr)


# 로컬 파일 수집
# readr(read_OOO)

read_csv("c:/data/team_standings.csv") # comma-separated
read_csv2("team_standings.csv") # semicolon-separated
read_tsv() # tab_separated
read_delim("team_standings.csv", delim=" ")
# General function(범용함수) delimiter(" ", ",")를 지정해서 분리할 수 있다.
read_fwf() # fixed width files
read_log() # log files




# 원격 웹서버 파일 수집
# readr(read_OOO)

# 원격 웹서버에 csv나 tsv파일이 있는 경우를 'flat files' 라고 한다.

read_csv("URL 주소/디렉토리명/파일명")

zika_data <- read_csv("https://~~~~/2016-06-25.csv")

View(zika_data)


