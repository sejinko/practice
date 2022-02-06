# 매트릭스 생성하기1

m <- matrix(nrow=2, ncol = 3)
m

# 매트릭스 생성하기2
m <- 1:10
m

dim(m) <- c(2,5)
m

# 매트릭스 생성하기3
x <- 1:3
y <- 10:12
cbind(x, y)

rbind(x, y)

z <- matrix(1:20, 4, 5) # 데이터, nrow, ncol
z

z <- matrix(2,4,5)
z

z <- matrix(c(1,2,3,4,5,6), nrow=2, ncol=3, byrow=T)
# byrow를 TRUE로 하면 데이터가 row로 먼저 채워진다.
z

# 행렬 결합
x <- 1:4
y <- 5:8

cbind(x, y)
rbind(x, y)

B <- matrix(0, 4, 5)
B
cbind(B, 1:4)

A <- matrix(1:20, 4, 5)
B <- matrix(1:20, 4, 5)
C <- cbind(A, B)
C
# 매트릭스와 매트릭스를 cbind로 만들 수 있다.


# 행렬간 산술연산

A+B

A*B

A-B

A/B

# 행렬의 각 행과 열에 이름을 부여하기

z <- matrix(1:20, 4, 5)
z

colnames(z) <- c("alpha", "beta", "gamma", "delta", "eps")
z
rownames(z) <- c("a", 'b','c','d')
z

# 행렬의 인덱싱

z[1,2]
z['a','alpha']
z[4,]
z[,4]

getwd()
setwd('f:/onedrive/2020/git/01_R/01_book')
getwd()
