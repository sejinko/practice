
# 데이터 프레임 생성하기
x <- data.frame(id = 1:4, name = c("kim", 'lee', 'park', 'choi'))
x




x <- c(100, 75, 80)
y <- c("A302043", "A302044", "A302045")
z <- data.frame(score=x, ID=y)
z

dat.1 <- data.frame(x=1:3, y=c('a', 'b', 'c'))
str(dat.1)
# data.frame 함수를 사용해서 데이터를 만들경우 string을 만들경우
# "자동으로 factor(범주형)으로 변한다."

dat_2 <- data.frame(x=1:3, y=c('a','b','c'), stringsAsFactors=F)
str(dat_2)
# stringsAsFactors의 디폴트가 TRUE로 되어 있다고 생각하면 된다. F로 하면
# factor로 되지 않고 charicter로 된다.




a <- data.frame(x=c(5,10,15), y=c("a", "b", "c"))
b <- data.frame(z=c(10,20,30))
cbind(a,b) # cbind는 data.frame에 row의 갯수가 같아야 사용할 수 있다.




a1 <- data.frame(x=c(20,25,30), y=c("d","e","f"))
rbind(a, a1) # rbind는 data.frame의 column의 개수가 같아야 사용할 수 있다.
