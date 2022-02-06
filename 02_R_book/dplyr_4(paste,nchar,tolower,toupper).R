# R에서 '처리'부분에서 필요한 패키지 종류
# 데이터 소스 → 수집 → 저장 → 처리 → 분석 → 표현(빅데이터 전처리 과정)


# 데이터 처리
# dplyr(paste/nchar/toupper/tolower)




# paste()

paste("Square", "circle", "Triangle") # 디폴트로 공백이 추가된다.
paste("Square", "circle", "Triangle", sep = "+")


shapes <- c("Square", "Circle", "Triangle")
shapes
paste("My favorite shape is a", shapes) # 3개의 문장이 만들어짐

two_cities <- c("best", "worst")
paste("It was the", two_cities, "of times.") # 2개의 문장이 만들어짐


paste(shapes, collapse=" ") # 공백으로 나눠지는 1개의 string으로 만들어짐


paste0("Square", 'Circle', 'Triangle')


# nchar(문자의 수를 셀때 사용)

nchar("Supercalifragilisticexpialidocious")


# tolower() & toupper()

cases <- c("CAPS", "low", "Title")
tolower(cases)

toupper(cases)
