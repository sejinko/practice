# R에서 '처리'부분에서 필요한 패키지 종류
# 데이터 소스 → 수집 → 저장 → 처리 → 분석 → 표현(빅데이터 전처리 과정)


# 데이터 처리
# dplyr(Functions for Regular Expression)

# 정규표현식 = 정규식 = Regular Expression = Regex
# 정규표현식은 패턴을 정의하는 string이다.



# grepl() : 패턴에 문자열이 들어 있느냐?
regular_expression <- "a"
string_to_search <- "Maryland"

grepl(regular_expression, string_to_search)
# 첫번째는 찾고자하는 패턴, 두번째는 문자열


grepl("land", "Maryland")

grepl("ryla", "Maryland")

grepl("dany", "Maryland")




# Meta_character : 다른 문자를 표현하는 문자

# period = .

grepl(".", "Maryland")
grepl(".", "*&[0+,%<@#~|}")
grepl(".", "")
grepl("a.b", c("aaa", "aab", "abb", "acadb"))
# a로 시작하고 중간에 어떤 문자가 있고 b로 끝나는게 있느냐?


# + : 하나 혹은 그 이상을 표현하는 것
grepl("a+", "Maryland") # a라는 것이 하나 이상 있어야 한다.
grepl("x+", "Maryland")


# * : 앞에 0 이상이 오느냐?
grepl("x*", "Maryland") # 0도 포함하기 때문에 없어도 TURE이다.


# ? : 


# {} = curly bracket(곱슬머리) : 안의 숫자만큼 와야 한다.
grepl("s{2}", "Mississippi") # s가 2번오는 패턴이 있느냐?
grepl("s{2,3}", "Mississippi") # ss or sss 패턴이 있느냐?
grepl("i{2,3}", "Mississippi") # ii or iii 패턴이 있느냐?

grepl("(iss){2}", "Mississippi") # ississ가 있어야 한다.
grepl("(ss){2}", "Mississippi") # ssss가 있어야 한다.
grepl("(i.{2}){3}", "Mississippi") # i~~i~~i~~이 있어야 한다.


# "^" (caret) : 문자열의 시작을 매칭(문자열의 시작은 이것으로 해)
grepl("^a", c("bab", "aab")) # a로 시작되니?


# "$" " 문자열의 끝을 매칭(문자열의 끝은 이것으로 해)
grepl("b$", c("bab", "aab")) # b로 끝나니?
grepl("^[ab]+$", c("bab","aab","abc"))
# ^[ab] : 문자열의 시작을 a or b로 시작하는 패턴이 있느냐?
# +$ : 앞에 있는 값인 a or b로 끝나는 패턴이 있느냐?


# or(|) : vertical bar로 읽으며 a라는 표현과 b라는 표현이 있느냐
grepl("a|b", c("abc", "bcd", "cde")) # a or b 가 있느냐?
grepl("North|South", c("South Dakota", "North Carolina", "West Virginia"))
# North or South가 있느냐"




# Character Sets(문자 집합)

# \\w(소문자w) : words를 표현함(letter, digit, underscore를 포함)
# \\d(소문자d) : digits를 표현한(0 부터 9까지)
# \\s(소문자s) : whitespace characters(공백문자)를 표현함(line breaks, tabs, spaces 포함)
# \\W(소문자W) : not words
# \\D(소문자D) : not digits
# \\S(소문자S) : not whitespace characters
grepl("\\w", "abcdefghijklmnopqrstuvwxyz0123456789") # word가 있느냐?
grepl("\\d", "0123456789") # digit이 있느냐?
grepl("\\s", "\n\t    ") # whitespace가 있느냐?


# [] = straight brackets(대괄호) - 직선으로 되어 있어서 스트레이트 브레이킷
# 대괄호안에 문자 셋을 사용할 수 있다.
grepl("[aeiou]", "rhythms") # aeiou 중에 하나라도 포함되는게 있느냐?
grepl("[^aeiou]", "rhythms") # aeiou 아닌 것중에 하나라도 포함되는게 있느냐?
grepl("[a-m]", "xyz") # a ~ m 범위에 있는 것중에 하나라도 포함되는게 있느냐?
grepl("[a-m]", "ABC")
grepl("[a-mA-M]", "ABC") # a ~ m, A ~ M 범위에 있는 것중에 하나라도 포함되는게 있느냐?




# Escape Character

# \\(two backslashes)
grepl('\\+', "tragedy + time = humor")
# +가 가지고 있는 의미를 없애겠다는 것 = 그냥 더하기라는 기호가 있느냐?

grepl('\\.', "http://www.jhsph.edu/")
# .이 가지고 있는 의미를 없애겠다는 것 = 그냥 점이라는 기호가 있느냐?




# grepl() = grep logical : 찾아서 있으면 TRUE,  없으면 FALSE로 리턴한다.
grepl("[Ii]", c("Hawaii", "Illiois", "Kentucky")) # 대문자 소문자 i가 있느냐?

# grep() : 찾아서 있으면 매칭이되면 해당 인덱스를 리턴한다.
grep("[Ii]", c("Hawaii", "Illinois", "Kentucky"))

# sub() = substitute(대체하다), replacement : 찾은것을 다른 것으로 변형해라
sub("[Ii]", "1", c("Hawaii", "Illinois", "Kentucky"))
# 첫번째 만난 대문자 I와 소문자 i를 찾아서 1로 바꿔라

# gsub() = global substitute : 
gsub("[Ii]", "1", c("Hawaii", "Illinois", "Kentucky"))
# 모든 대문자 I와 소문자 i를 찾아서 1로 바꿔라

# strsplit() : 문자열을 보고 분리해라
two_s <- state.name[grep("ss"), state.name)]
# ss를 state.name에서 찾아서 인덱스 값을 리턴해라
# state.name에서 그 인덱스 값에 해당하는 값을 찾아서 two_s에 넣어라.

strsplit(two_s, "ss") # two_s에서 ss를 찾아서 분리시켜라

