# 결측치 처리

# 아래와 같이 작업할때 R에서 작업할때 결측치가 생긴다
# 1. 인덱스 넘는 자리에 값을 넣을때
# 2. Explicit coercion 데이터타입을 변경할때
# 3. 매트릭스 생성할때 값을 안넣고 생성할때
# 4. Merging datasets 데이터 세트를 조인할떄 공통분모가 없을때

x <- c(1, 2, NA, 10, 3)
is.na(x) # NA(Not Available)이 있느냐? 하는 함수
is.nan(x) # NaN(Not a Number)가 있느냐? 하는 함수


x <- c(1,2, NaN, NA, 4)
is.na(x) # NaN과 NA 모두 TRUE로 표시한다.
is.nan(x)
!is.na(x)
!is.nan(x)


# NA 이외의 결측치 종류 = Inf(Infinite), -Inf, NaN(Not a Number)

x <- 1/0
x # Inf

x <- -1/0
x # -Inf

x <- 0/0
x # NaN


# 결측치 관련 함수

is.na(x)
is.nan(x)
!is.na(x)
!is.nan(x)
na.omit(x) # '데이터프레임'의 모든 컬럼의 결측치 모두 제거

mean(x$y, na.rm=T) # 결측치를 제외하고 y컬럼의 평균 계산
x$y <- ifelse(is.na(x$y), 55, x$y) 
# y컬럼의 평균을 결측치의 55(대체값)으로 교체함,
# 결측치가 없으면 그냥 그대로 가지고 있던 값을 써라 라는 뜻









