
# 벡터

x <- c(0.5, 0.6)
x

x <- c(T, F)
x

x <- 9:24
x

x <- c(1, 2.5, 3.2) # double로 python의 flout를 double로 보는 것 같다.
x

y <- c(1L, 2L, 3L) # interger
y

z <- c("ktx", 'saemaul', 'Mugunghwa') # string
z

v <- c(TRUE, FALSE, FALSE, TRUE) # logical
v

# 벡터에서 인덱싱

x[3] # R에서는 0번부터 세지 않는다.

x[c(1,3)]


# 벡터에 이름 부여하기

fruit <- c(5,3,2)
names(fruit) <- c('apple', 'orange', 'peach') # 컬럼명을 넣는 것 같다.
fruit

fruit <- setNames(c(5,3,2), c('apple', 'orange', 'peach'))
fruit


# 벡터의 길이 구하기
length(x)


# 기존 벡터 사용하여 새로운 벡터 생성하기

a <- c(1,2,3)
b <- c(5,6)
x <- c(a,4,b)
x

# 인덱스 범위가 넘는 자리에 값을 할당하면?
a[7] <- 2 # 중간에 NA가 나온다.
a

# 기존 벡터에 객체 삽입하기
x <- append(x, 99, after=3)
x
append(x,88, after=0) # 변수에 입력을 해주지 않으면 추가되지 않음
x


# seq(), rep()를 사용해서 벡터 생상하기(sequence, repeat)
x <- seq(from=0, to=1, by=0.1 )
x

y <- seq(from=0, to=1, length=11)
y

rep(1, 10)

#벡터간 산술 연산
x <- 1:3; y <- c(2,2,2)
x
y

x+y

x-y

x*y

x/y

x^y

# 벡터를 다른 클래스의 객체들이 오면? implicit coercion(알아서 변경됨)

y <- c(1.7, 'a') ## character
y

y <- c(TRUE, 2) ## numeric
y

y <- c('a', TRUE) ## character
y

# Logical < Numeric < Character


# 다른 형태의 객체로 변경하기(Explicit Coercion)
x <- 0:6
class(x)

as.numeric(x)

as.logical(x)

as.character(x)

x <- c("a", 'b','c')
as.numeric(x) # warning을 주고 미싱벨류를 만들어냄

as.complex(x)
