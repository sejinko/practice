# R에서 '처리'부분에서 필요한 패키지 종류
# 데이터 소스 → 수집 → 저장 → 처리 → 분석 → 표현(빅데이터 전처리 과정)


# 데이터 처리
# dplyr(Functions for Regular Expression)

# 정규표현식 = 정규식 = Regular Expression = Regex
# 정규표현식은 패턴을 정의하는 string이다.


#Character Class
# [[:alnum:]] = alphabetic, numeric
# [[:alpha:]] = only alphabetic
# [[:blank:]] = space, tab-space
# [[:cntrl:]] = line feed(줄바꿈), backspace, tabspace 제어문을 표현
# [[:digit:]] = digit 표현
# [[:graph:]] = alphabets, special characters, numbers
# [[:lower:]] = lower case alphabets(소문자만 표현)
# [[:upper:]] = uppercase alphabets(대문자만 표현)
# [[:print:]] = all printable characters like alphabhets, numbers
# [[:punct:]] = all punctuation characters(마침표, 느낌표)
# [[:space:]] = all space character like form feed, newline, carriage return, tab(공백문자)
# [[:xdigit:]] = hexadecimal digit character


# 정규 표현식의 자주 사용하는 예시

# 모든 공백 체크 정규식 :
# \\s
# 숫자만 체크 정규식 :
# ^[0-9]+$
# 이메일 체크 정규식 :
# ^[0-9a-zA-Z]([-_\\.]?[0-9a-zA-Z])*@[0-9a-zA-Z]([-_\\.]?[0-9a-zA-Z]*\\.[a-zA-Z]{2,3}$
# 핸드폰번호 정규식 :
# ^\\d{3}-\\d{3,4}-\\d{4}$
# 일반 전화번호 정규식 :
# ^\\d{2,3}-\\d{3,4}-\\d{4}$
# 아이디나 비밀번호 정규식 :
# ^[a-z0-9_]{4,20}$
# 휴대폰번호 체크 정규식 :
# ^01([0|1|6|7|8|9]?)-?([0-9]{3,4})-?([0-9]{4})$

