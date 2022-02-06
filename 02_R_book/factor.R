# 범주형 데이터(Factors)
# 범주형 데이터는 'self-describing': 어떤 데이터 자체가 그 값만 보더라도 스스로 표현할 수 있는 특성
는
# 정수로 표현하지 말고 라벨을 줘서 Factor로 표현하는 것이 좋다.
# 카테고리가 있는 데이터

x <- factor(c('yes', 'yes', 'no', 'yes', 'no')) 
x 

x <- factor(c('yes', 'yes', 'no', 'yes', 'no'), levels = c('yes', 'no')) #level(수준)을 만들 수 있다.
x # levels을 사용해서 yes가 no보다 수준이 높게 된다.

blood.type <- factor(c('A','A','AB','O','O'), levels=c('A','B','AB','O'))
table(blood.type)

blood.type <- factor(c('A','A','AB','O','O'), levels=c('O','B','A','AB'))
table(blood.type) #table을 사용하면 도수분포표가 나온다.

