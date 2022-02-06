# R에서 '처리'부분에서 필요한 패키지 종류
# 데이터 소스 → 수집 → 저장 → 처리 → 분석 → 표현(빅데이터 전처리 과정)


# 데이터 처리
# dplyr(OOOO_join)

library(dplyr)


# 데이터 병합

rbind() # row(column의 수가 같아야 한다.)
cbind() # column(row의 수가 같아야 한다.)

left_join() # 왼쪽 테이블 중심으로 병합(왼쪽 테이블은 없어지지 않는다.)
right_join() # 오른쪽 테이블 중심으로 병합
inner_join() # 왼쪽 테이블과 오른쪽 테이블의 교집합
full_join() # 왼쪽 테이블과 오른쪽 테이블의 합집합(outer join)




student <- data.frame(학번 = c(9154001, 9155001, 9156001),
                        이름 = c("홍길동", "김삿갓", "이영희"),
                        성별 = c("남", "남", "여"),
                        학과 = c("컴퓨터공", "전자공", "경영"),
                        학년 = c(4, 4, 3),
                        stringsAsFactors = FALSE)
str(student)


course <- data.frame(학번 = c(9154001, 9154001, 9154001, 9155001, 9155001),
                       과목명 = c("빅데이터", "운영체제",
                               "데이터베이스", "빅데이터", "운영체제"),
                       점수 = c(95, 88, 93, 88, 85),
                       stringsAsFactors = FALSE)
str(course)


student_course_left <- left_join(student, course, by="학번")
student_course_left

student_course_right <- right_join(student, course, by='학번')
student_course_right

student_course_inner <- inner_join(student, course, by='학번')
student_course_inner

student_course_full <- full_join(student, course, by='학번')
student_course_full


